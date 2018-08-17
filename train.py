from fastai.text import *
from model.crf_layer import CRF, log_sum_exp, init_linear, create_mask
import fire
"""
the fit and learner function below are rewrite by Emrys from the fastai library
"""

class Learner():
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None, clip=None, crit=None):
        """
        Combines a ModelData object with a nn.Module object, such that you can train that
        module.
        data (ModelData): An instance of ModelData.
        models(module): chosen neural architecture for solving a supported problem.
        opt_fn(function): optimizer function, uses SGD with Momentum of .9 if none.
        tmp_name(str): output name of the directory containing temporary files from training process
        models_name(str): output name of the directory containing the trained model
        metrics(list): array of functions for evaluating a desired metric. Eg. accuracy.
        clip(float): gradient clip chosen to limit the change in the gradient to prevent exploding gradients Eg. .3
        """
        self.data_,self.models,self.metrics = data,models,metrics
        self.sched=None
        self.wd_sched = None
        self.clip = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = tmp_name if os.path.isabs(tmp_name) else os.path.join(self.data.path, tmp_name)
        self.models_path = models_name if os.path.isabs(models_name) else os.path.join(self.data.path, models_name)
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit = crit if crit else self._get_crit(data)
        self.reg_fn = None
        self.fp16 = False

    @classmethod
    def from_model_data(cls, m, data, **kwargs):
        self = cls(data, BasicModel(to_gpu(m)), **kwargs)
        self.unfreeze()
        return self

    def __getitem__(self,i): return self.children[i]

    @property
    def children(self): return children(self.model)

    @property
    def model(self): return self.models.model

    @property
    def data(self): return self.data_

    def summary(self): return model_summary(self.model, [torch.rand(3, 3, self.data.sz,self.data.sz)])

    def __repr__(self): return self.model.__repr__()
    
    def lsuv_init(self, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False):         
        x = V(next(iter(self.data.trn_dl))[0])
        self.models.model=apply_lsuv_init(self.model, x, needed_std=needed_std, std_tol=std_tol,
                            max_attempts=max_attempts, do_orthonorm=do_orthonorm, 
                            cuda=USE_GPU and torch.cuda.is_available())

    def set_bn_freeze(self, m, do_freeze):
        if hasattr(m, 'running_mean'): m.bn_freeze = do_freeze

    def bn_freeze(self, do_freeze):
        apply_leaf(self.model, lambda m: self.set_bn_freeze(m, do_freeze))

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def freeze_all_but(self, n):
        c=self.get_layer_groups()
        for l in c: set_trainable(l, False)
        set_trainable(c[n], True)
        
    def freeze_groups(self, groups):
        c = self.get_layer_groups()
        self.unfreeze()
        for g in groups:
            set_trainable(c[g], False)
            
    def unfreeze_groups(self, groups):
        c = self.get_layer_groups()
        for g in groups:
            set_trainable(c[g], True)

    def unfreeze(self): self.freeze_to(0)

    def get_model_path(self, name): return os.path.join(self.models_path,name)+'.h5'
    
    def save(self, name): 
        save_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'): save_model(self.swa_model, self.get_model_path(name)[:-3]+'-swa.h5')
                       
    def load(self, name): 
        load_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'): load_model(self.swa_model, self.get_model_path(name)[:-3]+'-swa.h5')

    def set_data(self, data): self.data_ = data

    def get_cycle_end(self, name):
        if name is None: return None
        return lambda sched, cycle: self.save_cycle(name, cycle)

    def save_cycle(self, name, cycle): self.save(f'{name}_cyc_{cycle}')
    def load_cycle(self, name, cycle): self.load(f'{name}_cyc_{cycle}')

    def half(self):
        if self.fp16: return
        self.fp16 = True
        if type(self.model) != FP16: self.models.model = FP16(self.model)
    def float(self):
        if not self.fp16: return
        self.fp16 = False
        if type(self.model) == FP16: self.models.model = self.model.module
        self.model.float()

    def fit_gen(self, model, data, layer_opt, n_cycle, cycle_len=None, cycle_mult=1, cycle_save_name=None, best_save_name=None,
                use_clr=None, use_clr_beta=None, metrics=None, callbacks=None, use_wd_sched=False, norm_wds=False,             
                wds_sched_mult=None, use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):

        """Method does some preparation before finally delegating to the 'fit' method for
        fitting the model. Namely, if cycle_len is defined, it adds a 'Cosine Annealing'
        scheduler for varying the learning rate across iterations.

        Method also computes the total number of epochs to fit based on provided 'cycle_len',
        'cycle_mult', and 'n_cycle' parameters.

        Args:
            model (Learner):  Any neural architecture for solving a supported problem.
                Eg. ResNet-34, RNN_Learner etc.

            data (ModelData): An instance of ModelData.

            layer_opt (LayerOptimizer): An instance of the LayerOptimizer class

            n_cycle (int): number of cycles

            cycle_len (int):  number of cycles before lr is reset to the initial value.
                E.g if cycle_len = 3, then the lr is varied between a maximum
                and minimum value over 3 epochs.

            cycle_mult (int): additional parameter for influencing how the lr resets over
                the cycles. For an intuitive explanation, please see
                https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb

            cycle_save_name (str): use to save the weights at end of each cycle (requires
                use_clr, use_clr_beta or cycle_len arg)

            best_save_name (str): use to save weights of best model during training.

            metrics (function): some function for evaluating a desired metric. Eg. accuracy.

            callbacks (list(Callback)): callbacks to apply during the training.

            use_wd_sched (bool, optional): set to True to enable weight regularization using
                the technique mentioned in https://arxiv.org/abs/1711.05101. When this is True
                alone (see below), the regularization is detached from gradient update and
                applied directly to the weights.

            norm_wds (bool, optional): when this is set to True along with use_wd_sched, the
                regularization factor is normalized with each training cycle.

            wds_sched_mult (function, optional): when this is provided along with use_wd_sched
                as True, the value computed by this function is multiplied with the regularization
                strength. This function is passed the WeightDecaySchedule object. And example
                function that can be passed is:
                            f = lambda x: np.array(x.layer_opt.lrs) / x.init_lrs
                            
            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the 
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work.
                
            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.
                
            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.

        Returns:
            None
        """

        if cycle_save_name:
            assert use_clr or use_clr_beta or cycle_len, "cycle_save_name argument requires either of the following arguments use_clr, use_clr_beta, cycle_len"

        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics

        if use_wd_sched:
            # This needs to come before CosAnneal() because we need to read the initial learning rate from
            # layer_opt.lrs - but CosAnneal() alters the layer_opt.lrs value initially (divides by 100)
            if np.sum(layer_opt.wds) == 0:
                print('fit() warning: use_wd_sched is set to True, but weight decay(s) passed are 0. Use wds to '
                      'pass weight decay values.')
            batch_per_epoch = len(data.trn_dl)
            cl = cycle_len if cycle_len else 1
            self.wd_sched = WeightDecaySchedule(layer_opt, batch_per_epoch, cl, cycle_mult, n_cycle,
                                                norm_wds, wds_sched_mult)
            callbacks += [self.wd_sched]

        if use_clr is not None:
            clr_div,cut_div = use_clr[:2]
            moms = use_clr[2:] if len(use_clr) > 2 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            assert cycle_len, "use_clr requires cycle_len arg"
            self.sched = CircularLR(layer_opt, len(data.trn_dl)*cycle_len, on_cycle_end=cycle_end, div=clr_div, cut_div=cut_div,
                                    momentums=moms)
        elif use_clr_beta is not None:
            div,pct = use_clr_beta[:2]
            moms = use_clr_beta[2:] if len(use_clr_beta) > 3 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            assert cycle_len, "use_clr_beta requires cycle_len arg"
            self.sched = CircularLR_beta(layer_opt, len(data.trn_dl)*cycle_len, on_cycle_end=cycle_end, div=div,
                                    pct=pct, momentums=moms)
        elif cycle_len:
            cycle_end = self.get_cycle_end(cycle_save_name)
            cycle_batches = len(data.trn_dl)*cycle_len
            self.sched = CosAnneal(layer_opt, cycle_batches, on_cycle_end=cycle_end, cycle_mult=cycle_mult)
        elif not self.sched: self.sched=LossRecorder(layer_opt)
        callbacks+=[self.sched]

        if best_save_name is not None:
            callbacks+=[SaveBestModel(self, layer_opt, metrics, best_save_name)]

        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(model)
            callbacks+=[SWA(model, self.swa_model, swa_start)]

        n_epoch = int(sum_geom(cycle_len if cycle_len else 1, cycle_mult, n_cycle))
        # Emrys added the self.crf model
        return fit(model, self.crf_model, data, n_epoch, layer_opt.opt, self.crit,
            metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
            swa_model=self.swa_model if use_swa else None, swa_start=swa_start, 
            swa_eval_freq=swa_eval_freq, **kwargs)

    def get_layer_groups(self): return self.models.get_layer_groups()

    def get_layer_opt(self, lrs, wds):

        """Method returns an instance of the LayerOptimizer class, which
        allows for setting differential learning rates for different
        parts of the model.

        An example of how a model maybe differentiated into different parts
        for application of differential learning rates and weight decays is
        seen in ../.../courses/dl1/fastai/conv_learner.py, using the dict
        'model_meta'. Currently, this seems supported only for convolutional
        networks such as VGG-19, ResNet-XX etc.

        Args:
            lrs (float or list(float)): learning rate(s) for the model

            wds (float or list(float)): weight decay parameter(s).

        Returns:
            An instance of a LayerOptimizer
        """
        return LayerOptimizer(self.opt_fn, self.get_layer_groups(), lrs, wds)

    def fit(self, lrs, n_cycle, wds=None, **kwargs):

        """Method gets an instance of LayerOptimizer and delegates to self.fit_gen(..)

        Note that one can specify a list of learning rates which, when appropriately
        defined, will be applied to different segments of an architecture. This seems
        mostly relevant to ImageNet-trained models, where we want to alter the layers
        closest to the images by much smaller amounts.

        Likewise, a single or list of weight decay parameters can be specified, which
        if appropriate for a model, will apply variable weight decay parameters to
        different segments of the model.

        Args:
            lrs (float or list(float)): learning rate for the model

            n_cycle (int): number of cycles (or iterations) to fit the model for

            wds (float or list(float)): weight decay parameter(s).

            kwargs: other arguments

        Returns:
            None
        """
        self.sched = None
        layer_opt = self.get_layer_opt(lrs, wds)
        return self.fit_gen(self.model, self.data, layer_opt, n_cycle, **kwargs)

    def warm_up(self, lr, wds=None):
        layer_opt = self.get_layer_opt(lr/4, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), lr, linear=True)
        return self.fit_gen(self.model, self.data, layer_opt, 1)

    def lr_find(self, start_lr=1e-5, end_lr=10, wds=None, linear=False, **kwargs):
        """Helps you find an optimal learning rate for a model.

         It uses the technique developed in the 2015 paper
         `Cyclical Learning Rates for Training Neural Networks`, where
         we simply keep increasing the learning rate from a very small value,
         until the loss starts decreasing.

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            wds (iterable/float)

        Examples:
            As training moves us closer to the optimal weights for a model,
            the optimal learning rate will be smaller. We can take advantage of
            that knowledge and provide lr_find() with a starting learning rate
            1000x smaller than the model's current learning rate as such:

            >> learn.lr_find(lr/1000)

            >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
            >> learn.lr_find(lrs / 1000)

        Notes:
            lr_find() may finish before going through each batch of examples if
            the loss decreases enough.

        .. _Cyclical Learning Rates for Training Neural Networks:
            http://arxiv.org/abs/1506.01186

        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr, linear=linear)
        self.fit_gen(self.model, self.data, layer_opt, 1, **kwargs)
        self.load('tmp')

    def lr_find2(self, start_lr=1e-5, end_lr=10, num_it = 100, wds=None, linear=False, stop_dv=True, **kwargs):
        """A variant of lr_find() that helps find the best learning rate. It doesn't do
        an epoch but a fixed num of iterations (which may be more or less than an epoch
        depending on your data).
        At each step, it computes the validation loss and the metrics on the next
        batch of the validation data, so it's slower than lr_find().

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            num_it : the number of iterations you want it to run
            wds (iterable/float)
            stop_dv : stops (or not) when the losses starts to explode.
        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder2(layer_opt, num_it, end_lr, linear=linear, metrics=self.metrics, stop_dv=stop_dv)
        self.fit_gen(self.model, self.data, layer_opt, num_it//len(self.data.trn_dl) + 1, all_val=True, **kwargs)
        self.load('tmp')

    def predict(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict(m, dl)

    def predict_with_targs(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict_with_targs(m, dl)

    def predict_dl(self, dl): return predict_with_targs(self.model, dl)[0]

    def predict_array(self, arr):
        self.model.eval()
        return to_np(self.model(to_gpu(V(T(arr)))))

    def TTA(self, n_aug=4, is_test=False):
        """ Predict with Test Time Augmentation (TTA)

        Additional to the original test/validation images, apply image augmentation to them
        (just like for training images) and calculate the mean of predictions. The intent
        is to increase the accuracy of predictions by examining the images using multiple
        perspectives.

        Args:
            n_aug: a number of augmentation images to use per original image
            is_test: indicate to use test images; otherwise use validation images

        Returns:
            (tuple): a tuple containing:

                log predictions (numpy.ndarray): log predictions (i.e. `np.exp(log_preds)` will return probabilities)
                targs (numpy.ndarray): target values when `is_test==False`; zeros otherwise.
        """
        dl1 = self.data.test_dl     if is_test else self.data.val_dl
        dl2 = self.data.test_aug_dl if is_test else self.data.aug_dl
        preds1,targs = predict_with_targs(self.model, dl1)
        preds1 = [preds1]*math.ceil(n_aug/4)
        preds2 = [predict_with_targs(self.model, dl2)[0] for i in tqdm(range(n_aug), leave=False)]
        return np.stack(preds1+preds2), targs

    def fit_opt_sched(self, phases, cycle_save_name=None, best_save_name=None, stop_div=False, data_list=None, callbacks=None, 
                      cut = None, use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):
        """Wraps us the content of phases to send them to model.fit(..)

        This will split the training in several parts, each with their own learning rates/
        wds/momentums/optimizer detailed in phases.

        Additionaly we can add a list of different data objets in data_list to train
        on different datasets (to change the size for instance) for each of these groups.

        Args:
            phases: a list of TrainingPhase objects
            stop_div: when True, stops the training if the loss goes too high
            data_list: a list of different Data objects.
            kwargs: other arguments
            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the 
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work. 
            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.
            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.
        Returns:
            None
        """
        if data_list is None: data_list=[]
        if callbacks is None: callbacks=[]
        layer_opt = LayerOptimizer(phases[0].opt_fn, self.get_layer_groups(), 1e-2, phases[0].wds)
        if len(data_list) == 0: nb_batches = [len(self.data.trn_dl)] * len(phases)
        else: nb_batches = [len(data.trn_dl) for data in data_list] 
        self.sched = OptimScheduler(layer_opt, phases, nb_batches, stop_div)
        callbacks.append(self.sched)
        metrics = self.metrics
        if best_save_name is not None:
            callbacks+=[SaveBestModel(self, layer_opt, metrics, best_save_name)]
        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(self.model)
            callbacks+=[SWA(self.model, self.swa_model, swa_start)]
        n_epochs = [phase.epochs for phase in phases] if cut is None else cut
        if len(data_list)==0: data_list = [self.data]
        return fit(self.model, data_list, n_epochs,layer_opt, self.crit,
            metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
            swa_model=self.swa_model if use_swa else None, swa_start=swa_start, 
            swa_eval_freq=swa_eval_freq, **kwargs)

    def _get_crit(self, data): return F.mse_loss
    
# Emrys added the extra model at top
def fit(model, crf_model, data, n_epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper,
        swa_model=None, swa_start=None, swa_eval_freq=None, visualize=False, **kwargs):
    """ Fits a model

    Arguments:
       model (model): any pytorch module
           net = to_gpu(net)
       data (ModelData): see ModelData class and subclasses (can be a list)
       opts: an optimizer. Example: optim.Adam. 
       If n_epochs is a list, it needs to be the layer_optimizer to get the optimizer as it changes.
       n_epochs(int or list): number of epochs (or list of number of epochs)
       crit: loss function to optimize. Example: F.cross_entropy
    """

    seq_first = kwargs.pop('seq_first', False)
    all_val = kwargs.pop('all_val', False)
    get_ep_vals = kwargs.pop('get_ep_vals', False)
    validate_skip = kwargs.pop('validate_skip', 0)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    batch_num,avg_loss=0,0.
    for cb in callbacks: cb.on_train_begin()
    names = ["epoch", "trn_loss", "val_loss"] + [f.__name__ for f in metrics]
    if swa_model is not None:
        swa_names = ['swa_loss'] + [f'swa_{f.__name__}' for f in metrics]
        names += swa_names
        # will use this to call evaluate later
        swa_stepper = stepper(swa_model, None, crit, **kwargs)

    layout = "{!s:10} " * len(names)
    if not isinstance(n_epochs, Iterable): n_epochs=[n_epochs]
    if not isinstance(data, Iterable): data = [data]
    if len(data) == 1: data = data * len(n_epochs)
    for cb in callbacks: cb.on_phase_begin()
    model_stepper = stepper(model, crf_model, opt.opt if hasattr(opt,'opt') else opt, crit, **kwargs)
    ep_vals = collections.OrderedDict()
    tot_epochs = int(np.ceil(np.array(n_epochs).sum()))
    cnt_phases = np.array([ep * len(dat.trn_dl) for (ep,dat) in zip(n_epochs,data)]).cumsum()
    phase = 0
    for epoch in tnrange(tot_epochs, desc='Epoch'):
        if phase >= len(n_epochs): break #Sometimes cumulated errors make this append.
        model_stepper.reset(True)
        cur_data = data[phase]
        if hasattr(cur_data, 'trn_sampler'): cur_data.trn_sampler.set_epoch(epoch)
        if hasattr(cur_data, 'val_sampler'): cur_data.val_sampler.set_epoch(epoch)
        num_batch = len(cur_data.trn_dl)
        t = tqdm(iter(cur_data.trn_dl), leave=False, total=num_batch, miniters=0)
        if all_val: val_iter = IterBatch(cur_data.val_dl)

        for (*x,y) in t:
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = model_stepper.step(V(x),V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss, refresh=False)
            stop=False
            los = debias_loss if not all_val else [debias_loss] + validate_next(model_stepper,metrics, val_iter)
            for cb in callbacks: stop = stop or cb.on_batch_end(los)
            if stop: return
            if batch_num >= cnt_phases[phase]:
                for cb in callbacks: cb.on_phase_end()
                phase += 1
                if phase >= len(n_epochs):
                    t.close()
                    break
                for cb in callbacks: cb.on_phase_begin()
                if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
                if cur_data != data[phase]:
                    t.close()
                    break

        if not all_val:
            vals = validate(model_stepper, cur_data.val_dl, metrics, epoch, seq_first=seq_first, validate_skip = validate_skip)
            stop=False
            for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
            if swa_model is not None:
                if (epoch + 1) >= swa_start and ((epoch + 1 - swa_start) % swa_eval_freq == 0 or epoch == tot_epochs - 1):
                    fix_batchnorm(swa_model, cur_data.trn_dl)
                    swa_vals = validate(swa_stepper, cur_data.val_dl, metrics, epoch, validate_skip = validate_skip)
                    vals += swa_vals

            if epoch > 0: 
                print_stats(epoch, [debias_loss] + vals, visualize, prev_val)
            else:
                print(layout.format(*names))
                print_stats(epoch, [debias_loss] + vals, visualize)
            prev_val = [debias_loss] + vals
            ep_vals = append_stats(ep_vals, epoch, [debias_loss] + vals)
        if stop: break
    for cb in callbacks: cb.on_train_end()
    if get_ep_vals: return vals, ep_vals
    else: return vals

class Stepper():
    def __init__(self, m, crf_model, opt, crit, clip=0, reg_fn=None, fp16=False, loss_scale=1):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m,opt,crit,clip,reg_fn
        self.fp16 = fp16
        self.reset(True)
        if self.fp16: self.fp32_params = copy_model_to_fp32(m, opt)
        self.loss_scale = loss_scale
        self.crf_model = crf_model

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'):
            self.m.reset()
            if self.fp16: self.fp32_params = copy_model_to_fp32(self.m, self.opt)

    def step(self, xs, y, epoch):
        xtra = []
        output = self.m(*xs)
        if isinstance(output,tuple): 
            output,*xtra = output
        if self.fp16: self.m.zero_grad()
        else: self.opt.zero_grad()
        
        seq_len, bat_size = output.size()[0], output.size()[1]
        y = y.view(bat_size, seq_len, -1).permute(1,0,2)
        mask = create_mask(y, 1)
        
        loss = raw_loss = self.crit(output, y, mask)
        if self.loss_scale != 1: assert(self.fp16); loss = loss*self.loss_scale
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.fp16: update_fp32_grads(self.fp32_params, self.m)
        if self.loss_scale != 1:
            for param in self.fp32_params: param.grad.data.div_(self.loss_scale)
        if self.clip:   # Gradient clipping
            if IS_TORCH_04: nn.utils.clip_grad_norm_(trainable_params_(self.m), self.clip)
            else: nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        if 'wd' in self.opt.param_groups[0] and self.opt.param_groups[0]['wd'] != 0: 
            #Weight decay out of the loss. After the gradient computation but before the step.
            for group in self.opt.param_groups:
                lr, wd = group['lr'], group['wd']
                for p in group['params']:
                    if p.grad is not None: p.data = p.data.add(-wd * lr, p.data)
        self.opt.step()
        if self.fp16: 
            copy_fp32_to_model(self.m, self.fp32_params)
            torch.cuda.synchronize()
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        preds_raw = self.m(*xs)
        if isinstance(preds_raw, tuple): 
            preds_raw=preds_raw[0] # preds [sl, batch_size, len(tagset)]
        seq_len, bat_size = preds_raw.size()[0], preds_raw.size()[1]
        y = y.view(bat_size, seq_len, -1).permute(1,0,2)
        mask = create_mask(y, 1)
        preds = self.crf_model.decode(preds_raw, mask)
        return preds, self.crit(preds_raw, y, mask)

def validate(stepper, dl, metrics, epoch, seq_first=False, validate_skip = 0):
    if epoch < validate_skip: return [float('nan')] + [float('nan')] * len(metrics)
    batch_cnts,loss,res = [],[],[]
    stepper.reset(False)
    with no_grad_context():
        for (*x,y) in iter(dl):
            y = VV(y)
            preds, l = stepper.evaluate(VV(x), y)     
            batch_cnts.append(batch_sz(x, seq_first=seq_first))
            loss.append(to_np(l))
            res.append([f(datafy(preds), datafy(y)) for f in metrics])
    return [np.average(loss, 0, weights=batch_cnts)] + list(np.average(np.stack(res), 0, weights=batch_cnts))

def accuracy(preds, targs):
    sl, bs = preds.size()
    preds = preds.view(-1)
    targs = targs.view(bs, sl+1)[:, 1:].permute(1, 0).contiguous().view(-1)
    return (preds==targs).float().mean()

class RNN_Encoder(nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and LSTM/QRNN layers

        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange=0.1

    def __init__(self, ntoken, emb_sz, n_hid, n_layers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, qrnn=False):
        """ Default constructor for the RNN_Encoder class

            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                n_hid (int): number of hidden activation per LSTM layer
                n_layers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.

            Returns:
                None
        """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs, self.qrnn = 1, qrnn
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        if self.qrnn:
            #Using QRNN requires cupy: https://github.com/cupy/cupy
            from .torchqrnn.qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(n_layers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.linear = WeightDrop(rnn.linear, wdrop, weights=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                1, bidirectional=bidir) for l in range(n_layers)]
            if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz,self.n_hid,self.n_layers,self.dropoute = emb_sz,n_hid,n_layers,dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(n_layers)])
        self.reset()
        
    def forward(self, input):
        """ 
        Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        with set_grad_enabled(self.training):
            emb = self.encoder_with_dropout(input, dropout=self.dropoute if self.training else 0)
            emb = self.dropouti(emb)
            raw_output = emb
            new_hidden,raw_outputs,outputs = [],[],[]
            for l, (rnn,drop) in enumerate(zip(self.rnns, self.dropouths)):
                current_input = raw_output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1: raw_output = drop(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

class MultiBatchSeqRNN(RNN_Encoder):
    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl,bs = input.size()
        for l in self.hidden:
            for h in l: h.data.zero_()
        # raw_outputs, outputs = [],[]
        raw_outputs, outputs = super().forward(input)
        # for i in range(0, sl, self.bptt):
        #     r, o = super().forward(input[i: min(i+self.bptt, sl)])
        #     if i>(sl-self.max_seq):
        #         raw_outputs.append(r)
        #         outputs.append(o)
        # return self.concat(raw_outputs), self.concat(outputs)
        return raw_outputs, outputs
    def one_hidden(self, l):
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        if IS_TORCH_04: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        if self.qrnn: [r.reset() for r in self.rnns]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self.one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.n_layers)]

            
class PoolingLinearClassifier(nn.Module):
    """
    this layer is rewrite by Emrys
    """
    def __init__(self, layers, drops):
        super().__init__()
        print(layers)
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])
    
    def forward(self, input):
        raw_outputs, outputs = input
        output = outputs[-1]
        bs,sl,_ = output.size()
        # x.size() = [sequence_length, batch_size, n_hidden] --> [sequence_length*batch_size, n_hidden]
        x = output.view(-1, _)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        l_x = l_x.view(bs, sl, -1)
        l_x = l_x.permute(1, 0, 2)
        return l_x, raw_outputs, outputs

class TextModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [(m.encoder, m.dropouti), *zip(m.rnns, m.dropouths), (self.model[1])]

class RNN_Learner(Learner):
    def __init__(self, data, models, crf_model, **kwargs):
        self.crf_model = crf_model
        super().__init__(data, models, **kwargs)
        
    # _get_crit function have been rewrite by Emrys
    def _get_crit(self, data): 
        """this class needs to be over written"""
        return self.crf_model

    def fit(self, *args, **kwargs): return super().fit(*args, **kwargs, seq_first=True)

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))
    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))

def get_rnn_classifier_seq(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, qrnn=False):
    rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
    crf = to_gpu(CRF(2,1,n_class,n_class))
    return SequentialRNN(rnn_enc, PoolingLinearClassifier(layers, drops)), crf

"""build the data"""
class SeqDataLoader(DataLoader):
    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        # res = self.np_collate([self.dataset[i] for i in indices], self.pad_idx)
        # if not self.transpose: return res
        # res[0] = res[0].T
        # print('First seq:', res[0][0])
        # print('First labels:', res[1][0])
        res[1] = np.reshape(res[1], -1)  # reshape the labels to one sequence
        return res


class TextSeqDataset(Dataset):
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x,self.y,self.backwards,self.sos,self.eos = x,y,backwards,sos,eos

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]  # we need to get y as array
        if self.backwards: x = list(reversed(x))
        if self.eos is not None: x = x + [self.eos]
        if self.sos is not None: x = [self.sos]+x
        return np.array(x),np.array(y)

    def __len__(self): return len(self.x)

"""repeat the functions again"""
class Learner():
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None, clip=None, crit=None):
        """
        Combines a ModelData object with a nn.Module object, such that you can train that
        module.
        data (ModelData): An instance of ModelData.
        models(module): chosen neural architecture for solving a supported problem.
        opt_fn(function): optimizer function, uses SGD with Momentum of .9 if none.
        tmp_name(str): output name of the directory containing temporary files from training process
        models_name(str): output name of the directory containing the trained model
        metrics(list): array of functions for evaluating a desired metric. Eg. accuracy.
        clip(float): gradient clip chosen to limit the change in the gradient to prevent exploding gradients Eg. .3
        """
        self.data_,self.models,self.metrics = data,models,metrics
        self.sched=None
        self.wd_sched = None
        self.clip = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = tmp_name if os.path.isabs(tmp_name) else os.path.join(self.data.path, tmp_name)
        self.models_path = models_name if os.path.isabs(models_name) else os.path.join(self.data.path, models_name)
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit = crit if crit else self._get_crit(data)
        self.reg_fn = None
        self.fp16 = False

    @classmethod
    def from_model_data(cls, m, data, **kwargs):
        self = cls(data, BasicModel(to_gpu(m)), **kwargs)
        self.unfreeze()
        return self

    def __getitem__(self,i): return self.children[i]

    @property
    def children(self): return children(self.model)

    @property
    def model(self): return self.models.model

    @property
    def data(self): return self.data_

    def summary(self): return model_summary(self.model, [torch.rand(3, 3, self.data.sz,self.data.sz)])

    def __repr__(self): return self.model.__repr__()
    
    def lsuv_init(self, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False):         
        x = V(next(iter(self.data.trn_dl))[0])
        self.models.model=apply_lsuv_init(self.model, x, needed_std=needed_std, std_tol=std_tol,
                            max_attempts=max_attempts, do_orthonorm=do_orthonorm, 
                            cuda=USE_GPU and torch.cuda.is_available())

    def set_bn_freeze(self, m, do_freeze):
        if hasattr(m, 'running_mean'): m.bn_freeze = do_freeze

    def bn_freeze(self, do_freeze):
        apply_leaf(self.model, lambda m: self.set_bn_freeze(m, do_freeze))

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def freeze_all_but(self, n):
        c=self.get_layer_groups()
        for l in c: set_trainable(l, False)
        set_trainable(c[n], True)
        
    def freeze_groups(self, groups):
        c = self.get_layer_groups()
        self.unfreeze()
        for g in groups:
            set_trainable(c[g], False)
            
    def unfreeze_groups(self, groups):
        c = self.get_layer_groups()
        for g in groups:
            set_trainable(c[g], True)

    def unfreeze(self): self.freeze_to(0)

    def get_model_path(self, name): return os.path.join(self.models_path,name)+'.h5'
    
    def save(self, name): 
        save_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'): save_model(self.swa_model, self.get_model_path(name)[:-3]+'-swa.h5')
                       
    def load(self, name): 
        load_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'): load_model(self.swa_model, self.get_model_path(name)[:-3]+'-swa.h5')

    def set_data(self, data): self.data_ = data

    def get_cycle_end(self, name):
        if name is None: return None
        return lambda sched, cycle: self.save_cycle(name, cycle)

    def save_cycle(self, name, cycle): self.save(f'{name}_cyc_{cycle}')
    def load_cycle(self, name, cycle): self.load(f'{name}_cyc_{cycle}')

    def half(self):
        if self.fp16: return
        self.fp16 = True
        if type(self.model) != FP16: self.models.model = FP16(self.model)
    def float(self):
        if not self.fp16: return
        self.fp16 = False
        if type(self.model) == FP16: self.models.model = self.model.module
        self.model.float()

    def fit_gen(self, model, data, layer_opt, n_cycle, cycle_len=None, cycle_mult=1, cycle_save_name=None, best_save_name=None,
                use_clr=None, use_clr_beta=None, metrics=None, callbacks=None, use_wd_sched=False, norm_wds=False,             
                wds_sched_mult=None, use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):

        """Method does some preparation before finally delegating to the 'fit' method for
        fitting the model. Namely, if cycle_len is defined, it adds a 'Cosine Annealing'
        scheduler for varying the learning rate across iterations.

        Method also computes the total number of epochs to fit based on provided 'cycle_len',
        'cycle_mult', and 'n_cycle' parameters.

        Args:
            model (Learner):  Any neural architecture for solving a supported problem.
                Eg. ResNet-34, RNN_Learner etc.

            data (ModelData): An instance of ModelData.

            layer_opt (LayerOptimizer): An instance of the LayerOptimizer class

            n_cycle (int): number of cycles

            cycle_len (int):  number of cycles before lr is reset to the initial value.
                E.g if cycle_len = 3, then the lr is varied between a maximum
                and minimum value over 3 epochs.

            cycle_mult (int): additional parameter for influencing how the lr resets over
                the cycles. For an intuitive explanation, please see
                https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb

            cycle_save_name (str): use to save the weights at end of each cycle (requires
                use_clr, use_clr_beta or cycle_len arg)

            best_save_name (str): use to save weights of best model during training.

            metrics (function): some function for evaluating a desired metric. Eg. accuracy.

            callbacks (list(Callback)): callbacks to apply during the training.

            use_wd_sched (bool, optional): set to True to enable weight regularization using
                the technique mentioned in https://arxiv.org/abs/1711.05101. When this is True
                alone (see below), the regularization is detached from gradient update and
                applied directly to the weights.

            norm_wds (bool, optional): when this is set to True along with use_wd_sched, the
                regularization factor is normalized with each training cycle.

            wds_sched_mult (function, optional): when this is provided along with use_wd_sched
                as True, the value computed by this function is multiplied with the regularization
                strength. This function is passed the WeightDecaySchedule object. And example
                function that can be passed is:
                            f = lambda x: np.array(x.layer_opt.lrs) / x.init_lrs
                            
            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the 
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work.
                
            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.
                
            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.

        Returns:
            None
        """

        if cycle_save_name:
            assert use_clr or use_clr_beta or cycle_len, "cycle_save_name argument requires either of the following arguments use_clr, use_clr_beta, cycle_len"

        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics

        if use_wd_sched:
            # This needs to come before CosAnneal() because we need to read the initial learning rate from
            # layer_opt.lrs - but CosAnneal() alters the layer_opt.lrs value initially (divides by 100)
            if np.sum(layer_opt.wds) == 0:
                print('fit() warning: use_wd_sched is set to True, but weight decay(s) passed are 0. Use wds to '
                      'pass weight decay values.')
            batch_per_epoch = len(data.trn_dl)
            cl = cycle_len if cycle_len else 1
            self.wd_sched = WeightDecaySchedule(layer_opt, batch_per_epoch, cl, cycle_mult, n_cycle,
                                                norm_wds, wds_sched_mult)
            callbacks += [self.wd_sched]

        if use_clr is not None:
            clr_div,cut_div = use_clr[:2]
            moms = use_clr[2:] if len(use_clr) > 2 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            assert cycle_len, "use_clr requires cycle_len arg"
            self.sched = CircularLR(layer_opt, len(data.trn_dl)*cycle_len, on_cycle_end=cycle_end, div=clr_div, cut_div=cut_div,
                                    momentums=moms)
        elif use_clr_beta is not None:
            div,pct = use_clr_beta[:2]
            moms = use_clr_beta[2:] if len(use_clr_beta) > 3 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            assert cycle_len, "use_clr_beta requires cycle_len arg"
            self.sched = CircularLR_beta(layer_opt, len(data.trn_dl)*cycle_len, on_cycle_end=cycle_end, div=div,
                                    pct=pct, momentums=moms)
        elif cycle_len:
            cycle_end = self.get_cycle_end(cycle_save_name)
            cycle_batches = len(data.trn_dl)*cycle_len
            self.sched = CosAnneal(layer_opt, cycle_batches, on_cycle_end=cycle_end, cycle_mult=cycle_mult)
        elif not self.sched: self.sched=LossRecorder(layer_opt)
        callbacks+=[self.sched]

        if best_save_name is not None:
            callbacks+=[SaveBestModel(self, layer_opt, metrics, best_save_name)]

        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(model)
            callbacks+=[SWA(model, self.swa_model, swa_start)]

        n_epoch = int(sum_geom(cycle_len if cycle_len else 1, cycle_mult, n_cycle))
        # Emrys added the self.crf model
        return fit(model, self.crf_model, data, n_epoch, layer_opt.opt, self.crit,
            metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
            swa_model=self.swa_model if use_swa else None, swa_start=swa_start, 
            swa_eval_freq=swa_eval_freq, **kwargs)

    def get_layer_groups(self): return self.models.get_layer_groups()

    def get_layer_opt(self, lrs, wds):

        """Method returns an instance of the LayerOptimizer class, which
        allows for setting differential learning rates for different
        parts of the model.

        An example of how a model maybe differentiated into different parts
        for application of differential learning rates and weight decays is
        seen in ../.../courses/dl1/fastai/conv_learner.py, using the dict
        'model_meta'. Currently, this seems supported only for convolutional
        networks such as VGG-19, ResNet-XX etc.

        Args:
            lrs (float or list(float)): learning rate(s) for the model

            wds (float or list(float)): weight decay parameter(s).

        Returns:
            An instance of a LayerOptimizer
        """
        return LayerOptimizer(self.opt_fn, self.get_layer_groups(), lrs, wds)

    def fit(self, lrs, n_cycle, wds=None, **kwargs):

        """Method gets an instance of LayerOptimizer and delegates to self.fit_gen(..)

        Note that one can specify a list of learning rates which, when appropriately
        defined, will be applied to different segments of an architecture. This seems
        mostly relevant to ImageNet-trained models, where we want to alter the layers
        closest to the images by much smaller amounts.

        Likewise, a single or list of weight decay parameters can be specified, which
        if appropriate for a model, will apply variable weight decay parameters to
        different segments of the model.

        Args:
            lrs (float or list(float)): learning rate for the model

            n_cycle (int): number of cycles (or iterations) to fit the model for

            wds (float or list(float)): weight decay parameter(s).

            kwargs: other arguments

        Returns:
            None
        """
        self.sched = None
        layer_opt = self.get_layer_opt(lrs, wds)
        return self.fit_gen(self.model, self.data, layer_opt, n_cycle, **kwargs)

    def warm_up(self, lr, wds=None):
        layer_opt = self.get_layer_opt(lr/4, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), lr, linear=True)
        return self.fit_gen(self.model, self.data, layer_opt, 1)

    def lr_find(self, start_lr=1e-5, end_lr=10, wds=None, linear=False, **kwargs):
        """Helps you find an optimal learning rate for a model.

         It uses the technique developed in the 2015 paper
         `Cyclical Learning Rates for Training Neural Networks`, where
         we simply keep increasing the learning rate from a very small value,
         until the loss starts decreasing.

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            wds (iterable/float)

        Examples:
            As training moves us closer to the optimal weights for a model,
            the optimal learning rate will be smaller. We can take advantage of
            that knowledge and provide lr_find() with a starting learning rate
            1000x smaller than the model's current learning rate as such:

            >> learn.lr_find(lr/1000)

            >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
            >> learn.lr_find(lrs / 1000)

        Notes:
            lr_find() may finish before going through each batch of examples if
            the loss decreases enough.

        .. _Cyclical Learning Rates for Training Neural Networks:
            http://arxiv.org/abs/1506.01186

        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr, linear=linear)
        self.fit_gen(self.model, self.data, layer_opt, 1, **kwargs)
        self.load('tmp')

    def lr_find2(self, start_lr=1e-5, end_lr=10, num_it = 100, wds=None, linear=False, stop_dv=True, **kwargs):
        """A variant of lr_find() that helps find the best learning rate. It doesn't do
        an epoch but a fixed num of iterations (which may be more or less than an epoch
        depending on your data).
        At each step, it computes the validation loss and the metrics on the next
        batch of the validation data, so it's slower than lr_find().

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            num_it : the number of iterations you want it to run
            wds (iterable/float)
            stop_dv : stops (or not) when the losses starts to explode.
        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder2(layer_opt, num_it, end_lr, linear=linear, metrics=self.metrics, stop_dv=stop_dv)
        self.fit_gen(self.model, self.data, layer_opt, num_it//len(self.data.trn_dl) + 1, all_val=True, **kwargs)
        self.load('tmp')

    def predict(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict(m, dl)

    def predict_with_targs(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict_with_targs(m, dl)

    def predict_dl(self, dl): return predict_with_targs(self.model, dl)[0]

    def predict_array(self, arr):
        self.model.eval()
        return to_np(self.model(to_gpu(V(T(arr)))))

    def TTA(self, n_aug=4, is_test=False):
        """ Predict with Test Time Augmentation (TTA)

        Additional to the original test/validation images, apply image augmentation to them
        (just like for training images) and calculate the mean of predictions. The intent
        is to increase the accuracy of predictions by examining the images using multiple
        perspectives.

        Args:
            n_aug: a number of augmentation images to use per original image
            is_test: indicate to use test images; otherwise use validation images

        Returns:
            (tuple): a tuple containing:

                log predictions (numpy.ndarray): log predictions (i.e. `np.exp(log_preds)` will return probabilities)
                targs (numpy.ndarray): target values when `is_test==False`; zeros otherwise.
        """
        dl1 = self.data.test_dl     if is_test else self.data.val_dl
        dl2 = self.data.test_aug_dl if is_test else self.data.aug_dl
        preds1,targs = predict_with_targs(self.model, dl1)
        preds1 = [preds1]*math.ceil(n_aug/4)
        preds2 = [predict_with_targs(self.model, dl2)[0] for i in tqdm(range(n_aug), leave=False)]
        return np.stack(preds1+preds2), targs

    def fit_opt_sched(self, phases, cycle_save_name=None, best_save_name=None, stop_div=False, data_list=None, callbacks=None, 
                      cut = None, use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):
        """Wraps us the content of phases to send them to model.fit(..)

        This will split the training in several parts, each with their own learning rates/
        wds/momentums/optimizer detailed in phases.

        Additionaly we can add a list of different data objets in data_list to train
        on different datasets (to change the size for instance) for each of these groups.

        Args:
            phases: a list of TrainingPhase objects
            stop_div: when True, stops the training if the loss goes too high
            data_list: a list of different Data objects.
            kwargs: other arguments
            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the 
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work. 
            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.
            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.
        Returns:
            None
        """
        if data_list is None: data_list=[]
        if callbacks is None: callbacks=[]
        layer_opt = LayerOptimizer(phases[0].opt_fn, self.get_layer_groups(), 1e-2, phases[0].wds)
        if len(data_list) == 0: nb_batches = [len(self.data.trn_dl)] * len(phases)
        else: nb_batches = [len(data.trn_dl) for data in data_list] 
        self.sched = OptimScheduler(layer_opt, phases, nb_batches, stop_div)
        callbacks.append(self.sched)
        metrics = self.metrics
        if best_save_name is not None:
            callbacks+=[SaveBestModel(self, layer_opt, metrics, best_save_name)]
        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(self.model)
            callbacks+=[SWA(self.model, self.swa_model, swa_start)]
        n_epochs = [phase.epochs for phase in phases] if cut is None else cut
        if len(data_list)==0: data_list = [self.data]
        return fit(self.model, data_list, n_epochs,layer_opt, self.crit,
            metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
            swa_model=self.swa_model if use_swa else None, swa_start=swa_start, 
            swa_eval_freq=swa_eval_freq, **kwargs)

    def _get_crit(self, data): return F.mse_loss
    
# Emrys added the extra model at top
def fit(model, crf_model, data, n_epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper,
        swa_model=None, swa_start=None, swa_eval_freq=None, visualize=False, **kwargs):
    """ Fits a model

    Arguments:
       model (model): any pytorch module
           net = to_gpu(net)
       data (ModelData): see ModelData class and subclasses (can be a list)
       opts: an optimizer. Example: optim.Adam. 
       If n_epochs is a list, it needs to be the layer_optimizer to get the optimizer as it changes.
       n_epochs(int or list): number of epochs (or list of number of epochs)
       crit: loss function to optimize. Example: F.cross_entropy
    """

    seq_first = kwargs.pop('seq_first', False)
    all_val = kwargs.pop('all_val', False)
    get_ep_vals = kwargs.pop('get_ep_vals', False)
    validate_skip = kwargs.pop('validate_skip', 0)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    batch_num,avg_loss=0,0.
    for cb in callbacks: cb.on_train_begin()
    names = ["epoch", "trn_loss", "val_loss"] + [f.__name__ for f in metrics]
    if swa_model is not None:
        swa_names = ['swa_loss'] + [f'swa_{f.__name__}' for f in metrics]
        names += swa_names
        # will use this to call evaluate later
        swa_stepper = stepper(swa_model, None, crit, **kwargs)

    layout = "{!s:10} " * len(names)
    if not isinstance(n_epochs, Iterable): n_epochs=[n_epochs]
    if not isinstance(data, Iterable): data = [data]
    if len(data) == 1: data = data * len(n_epochs)
    for cb in callbacks: cb.on_phase_begin()
    model_stepper = stepper(model, crf_model, opt.opt if hasattr(opt,'opt') else opt, crit, **kwargs)
    ep_vals = collections.OrderedDict()
    tot_epochs = int(np.ceil(np.array(n_epochs).sum()))
    cnt_phases = np.array([ep * len(dat.trn_dl) for (ep,dat) in zip(n_epochs,data)]).cumsum()
    phase = 0
    for epoch in tnrange(tot_epochs, desc='Epoch'):
        if phase >= len(n_epochs): break #Sometimes cumulated errors make this append.
        model_stepper.reset(True)
        cur_data = data[phase]
        if hasattr(cur_data, 'trn_sampler'): cur_data.trn_sampler.set_epoch(epoch)
        if hasattr(cur_data, 'val_sampler'): cur_data.val_sampler.set_epoch(epoch)
        num_batch = len(cur_data.trn_dl)
        t = tqdm(iter(cur_data.trn_dl), leave=False, total=num_batch, miniters=0)
        if all_val: val_iter = IterBatch(cur_data.val_dl)

        for (*x,y) in t:
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = model_stepper.step(V(x),V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss, refresh=False)
            stop=False
            los = debias_loss if not all_val else [debias_loss] + validate_next(model_stepper,metrics, val_iter)
            for cb in callbacks: stop = stop or cb.on_batch_end(los)
            if stop: return
            if batch_num >= cnt_phases[phase]:
                for cb in callbacks: cb.on_phase_end()
                phase += 1
                if phase >= len(n_epochs):
                    t.close()
                    break
                for cb in callbacks: cb.on_phase_begin()
                if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
                if cur_data != data[phase]:
                    t.close()
                    break

        if not all_val:
            vals = validate(model_stepper, cur_data.val_dl, metrics, epoch, seq_first=seq_first, validate_skip = validate_skip)
            stop=False
            for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
            if swa_model is not None:
                if (epoch + 1) >= swa_start and ((epoch + 1 - swa_start) % swa_eval_freq == 0 or epoch == tot_epochs - 1):
                    fix_batchnorm(swa_model, cur_data.trn_dl)
                    swa_vals = validate(swa_stepper, cur_data.val_dl, metrics, epoch, validate_skip = validate_skip)
                    vals += swa_vals

            if epoch > 0: 
                print_stats(epoch, [debias_loss] + vals, visualize, prev_val)
            else:
                print(layout.format(*names))
                print_stats(epoch, [debias_loss] + vals, visualize)
            prev_val = [debias_loss] + vals
            ep_vals = append_stats(ep_vals, epoch, [debias_loss] + vals)
        if stop: break
    for cb in callbacks: cb.on_train_end()
    if get_ep_vals: return vals, ep_vals
    else: return vals

class Stepper():
    def __init__(self, m, crf_model, opt, crit, clip=0, reg_fn=None, fp16=False, loss_scale=1):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m,opt,crit,clip,reg_fn
        self.fp16 = fp16
        self.reset(True)
        if self.fp16: self.fp32_params = copy_model_to_fp32(m, opt)
        self.loss_scale = loss_scale
        self.crf_model = crf_model

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'):
            self.m.reset()
            if self.fp16: self.fp32_params = copy_model_to_fp32(self.m, self.opt)

    def step(self, xs, y, epoch):
        xtra = []
        output = self.m(*xs)
        if isinstance(output,tuple): 
            output,*xtra = output
        if self.fp16: self.m.zero_grad()
        else: self.opt.zero_grad()
        
        seq_len, bat_size = output.size()[0], output.size()[1]
        y = y.view(bat_size, seq_len, -1).permute(1,0,2)
        mask = create_mask(y, 1)
        
        loss = raw_loss = self.crit(output, y, mask)
        if self.loss_scale != 1: assert(self.fp16); loss = loss*self.loss_scale
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.fp16: update_fp32_grads(self.fp32_params, self.m)
        if self.loss_scale != 1:
            for param in self.fp32_params: param.grad.data.div_(self.loss_scale)
        if self.clip:   # Gradient clipping
            if IS_TORCH_04: nn.utils.clip_grad_norm_(trainable_params_(self.m), self.clip)
            else: nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        if 'wd' in self.opt.param_groups[0] and self.opt.param_groups[0]['wd'] != 0: 
            #Weight decay out of the loss. After the gradient computation but before the step.
            for group in self.opt.param_groups:
                lr, wd = group['lr'], group['wd']
                for p in group['params']:
                    if p.grad is not None: p.data = p.data.add(-wd * lr, p.data)
        self.opt.step()
        if self.fp16: 
            copy_fp32_to_model(self.m, self.fp32_params)
            torch.cuda.synchronize()
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        preds_raw = self.m(*xs)
        if isinstance(preds_raw, tuple): 
            preds_raw=preds_raw[0] # preds [sl, batch_size, len(tagset)]
        seq_len, bat_size = preds_raw.size()[0], preds_raw.size()[1]
        y = y.view(bat_size, seq_len, -1).permute(1,0,2)
        mask = create_mask(y, 1)
        preds = self.crf_model.decode(preds_raw, mask)
        return preds, self.crit(preds_raw, y, mask)

def validate(stepper, dl, metrics, epoch, seq_first=False, validate_skip = 0):
    if epoch < validate_skip: return [float('nan')] + [float('nan')] * len(metrics)
    batch_cnts,loss,res = [],[],[]
    stepper.reset(False)
    with no_grad_context():
        for (*x,y) in iter(dl):
            y = VV(y)
            preds, l = stepper.evaluate(VV(x), y)     
            batch_cnts.append(batch_sz(x, seq_first=seq_first))
            loss.append(to_np(l))
            res.append([f(datafy(preds), datafy(y)) for f in metrics])
    return [np.average(loss, 0, weights=batch_cnts)] + list(np.average(np.stack(res), 0, weights=batch_cnts))

def accuracy(preds, targs):
    sl, bs = preds.size()
    preds = preds.view(-1)
    targs = targs.view(bs, sl+1)[:, 1:].permute(1, 0).contiguous().view(-1)
    return (preds==targs).float().mean()
""" finished repeating"""

def train(path, bptt=70, max_seq=20*70, em_sz=400, nh=1150, nl=3, ):
    """import data"""
    dir_path = Path(path)
    bs=64 # batch_size = 2
    trn_sent = np.load(dir_path / 'tmp' / f'trn_ids.npy')
    val_sent = np.load(dir_path / 'tmp' / f'val_ids.npy')
    test_sent = np.load(dir_path / 'tmp' / f'test_ids.npy')

    trn_lbls = np.load(dir_path / 'tmp' / f'lbl_trn.npy')
    val_lbls = np.load(dir_path / 'tmp' / f'lbl_val.npy')
    test_lbls = np.load(dir_path / 'tmp' / f'lbl_test.npy')
    id2label = pickle.load(open(dir_path / 'tmp' / 'itol.pkl', 'rb'))
    c = len(id2label)# for start tag and end tag



    id2token = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
    vs = len(id2token)

    print('Train sentences shape:', trn_sent.shape)
    print('Train labels shape:', trn_lbls.shape)
    print('Token ids:', [id2token[id_] for id_ in trn_sent[0]])
    print('Label ids:', [id2label[id_] for id_ in trn_lbls[0]])

    """ create dataloader"""
    trn_ds = TextSeqDataset(trn_sent, trn_lbls)
    val_ds = TextSeqDataset(val_sent, val_lbls)
    test_ds = TextSeqDataset(test_sent, test_lbls)
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs//2)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    test_samp = SortSampler(test_sent, key=lambda x: len(test_sent[x]))
    trn_dl = SeqDataLoader(trn_ds, bs//2, transpose=False, num_workers=1, pad_idx=1, sampler=trn_samp)  # TODO why transpose? Should we also transpose the labels?
    val_dl = SeqDataLoader(val_ds, bs, transpose=False, num_workers=1, pad_idx=1, sampler=val_samp)
    test_dl = SeqDataLoader(test_ds, bs, transpose=False, num_workers=1, pad_idx=1, sampler=test_samp)
    md = ModelData(dir_path, trn_dl, val_dl, test_dl)

    """build model"""
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5
    m, crf = get_rnn_classifier_seq(bptt, max_seq, c, vs, emb_sz=em_sz, 
    n_hid=nh, n_layers=nl, pad_token=1, layers=[em_sz, 50, c], drops=[dps[4], 0.1], dropouti=dps[0], 
    wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    learn = RNN_Learner(md, TextModel(to_gpu(m)), crf, opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip=25
    learn.metrics = [accuracy]

    lr=3e-5
    lrm = 2.6
    lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr]) 
    lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])
    wd = 1e-7
    wd = 0
    learn.load_encoder('lm1_enc')

    print('freeze to layer 1')
    learn.freeze_to(-1)
    
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
    learn.save('clas_0')
    learn.load('clas_0')
    
    learn.freeze_to(-2)
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))

    learn.save('clas_1')
    learn.load('clas_1')

    learn.unfreeze()
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(32,10))
    learn.save('clas_2')

if __name__ == '__main__':
    fire.Fire(train)
