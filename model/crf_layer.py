from fastai.text import *

def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def create_mask(y, pad_id):
    """
    Create mask for padding for CRF layer
    
    args:
        y (seq_len, bat_size) : the correct label for prediction
        pad_id       (scalar) : number for padding in id2token
    """
    sl, bs, _ = y.size()
    mask = (y != pad_id).view(sl, bs)
    return mask
    
class CRF(nn.Module):
    """
    Conditional Random Field (CRF) layer.
    """
    
    def __init__(self, start_tag, end_tag, hidden_dim, tagset_size):
        """
        args:
            start_tag  (scalar) : special start tag for CRF
            end_tag    (scalar) : special end tag for CRF
            hidden_dim (scalar) : input dim size
            tagset_size(scalar) : target_set_size
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size)
        self.rand_init()

    def rand_init(self):
        """
        random initialization
        """

        init_linear(self.hidden2tag)

    def cal_score(self, feats):
        """
        calculate CRF score
        :param feats (sentlen, batch_size, feature_num) : input features
        """

        sentlen = feats.size(0)
        batch_size = feats.size(1)
        crf_scores = self.hidden2tag(feats).view(-1, self.tagset_size, self.tagset_size)
        self.crf_scores = crf_scores.view(sentlen, batch_size, self.tagset_size, self.tagset_size)
        return self.crf_scores

    def forward(self, feats, target, mask):
        """
        calculate viterbi loss
        args:
            feats  (batch_size, seq_len, hidden_dim) : input features from word_rep layers
            target (batch_size, seq_len, 1) : crf label
            mask   (batch_size, seq_len) : mask for crf label
        """

        crf_scores = self.cal_score(feats)
        loss = self.get_loss(crf_scores, target, mask)
        return loss

    def get_loss(self, scores, target, mask):
        """
        calculate viterbi loss
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : class score for CRF
            target (seq_len, bat_size, 1) : crf label
            mask   (seq_len, bat_size) : mask for crf label
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.start_tag, :].clone()
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).\
                expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = log_sum_exp(cur_values, self.tagset_size)
            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx,
                                      cur_partition.masked_select(mask_idx))

        partition = partition[:, self.end_tag].sum()
        loss = (partition - tg_energy) / bat_size

        return loss + 10000

    def decode(self, feats, mask):
        """
        decode with dynamic programming
        args:
            feats (sentlen, batch_size, feature_num) : input features
            mask (seq_len, bat_size) : mask for padding
        """

        scores = self.cal_score(feats)
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = V(1 - mask.data, volatile=True)
        decode_idx = V(torch.cuda.LongTensor(seq_len-1, bat_size), volatile=True)

        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        forscores = inivalues[:, self.start_tag, :]
        back_points = list()
        for idx, cur_values in seq_iter:
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).\
                                                  expand(bat_size, self.tagset_size, self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer
        return decode_idx
