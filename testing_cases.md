prefix /fs-object-detection/paperspace/fastai/courses/coNLL/data/nlp_seq/ner/; cuda_id 0; lm_id ; clas_id None; bs 64; cl 1; 
backwards False; dropmult 1.0 unfreeze True startat 0; pretrain True; bpe False; use_clr True;use_regular_schedule False; use_discriminative True; last False;chain_thaw True; from_scratch True; train_file_id 
```
Train sentences shape: (14988,)
Train labels shape: (14988,)
Token ids: ['xbos', '-docstart-']
Label ids: ['_bos_', 'O']
Training classifier from scratch. LM encoder is not loaded.
Using chain-thaw. Unfreezing all layers one at a time...
# of layers: 5
Fine-tuning last layer...
Epoch
100% 1/1 [00:12<00:00, 12.50s/it]
epoch      trn_loss   val_loss   accuracy                   
    0      1.254827   1.164105   0.714305  

Fine-tuning layer #0.
Epoch
100% 1/1 [00:22<00:00, 22.51s/it]
epoch      trn_loss   val_loss   accuracy                   
    0      1.272519   1.121246   0.714305  

Fine-tuning layer #1.
Epoch
100% 1/1 [00:25<00:00, 25.27s/it]
epoch      trn_loss   val_loss   accuracy                    
    0      0.636733   0.588199   0.851106  

Fine-tuning layer #2.
Epoch
100% 1/1 [00:21<00:00, 21.86s/it]
epoch      trn_loss   val_loss   accuracy                    
    0      0.509855   0.519723   0.855141  

Fine-tuning layer #3.
Epoch
100% 1/1 [00:14<00:00, 14.38s/it]
epoch      trn_loss   val_loss   accuracy                    
    0      0.475224   0.503133   0.855224  

Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.
Epoch
100% 1/1 [00:31<00:00, 31.48s/it]
epoch      trn_loss   val_loss   accuracy                    
    0      0.35263    0.320918   0.896819  

Plotting lrs...
Validation f1 measure overall: 0.3585084232443195
{'precision-LOC': 0.6567083107007061, 'recall-LOC': 0.658138268916712, 'f1-measure-LOC': 0.6574225122348603, 'precision-MISC': 0.30539499036608864, 'recall-MISC': 0.34381778741865504, 'f1-measure-MISC': 0.32346938775505224, 'precision-PER': 0.20763723150357996, 'recall-PER': 0.18892508143322476, 'f1-measure-PER': 0.19783968163724402, 'precision-ORG': 0.2898550724637677, 'recall-ORG': 0.014914243102162566, 'f1-measure-ORG': 0.02836879432623183, 'precision-overall': 0.40960207612456745, 'recall-overall': 0.3187478963312016, 'f1-measure-overall': 0.3585084232443195}
Validation token-level accuracy of NER model: 0.8966.
Test f1 measure overall: 0.2860460206881511
{'precision-LOC': 0.6215066828675577, 'recall-LOC': 0.6133093525179856, 'f1-measure-LOC': 0.6173808086903544, 'precision-MISC': 0.20392584514721918, 'recall-MISC': 0.26638176638176636, 'f1-measure-MISC': 0.2310067943174308, 'precision-PER': 0.1018363939899833, 'recall-PER': 0.07544836116264687, 'f1-measure-PER': 0.08667850799284631, 'precision-ORG': 0.3538461538461533, 'recall-ORG': 0.013847080072245636, 'f1-measure-ORG': 0.0266512166859719, 'precision-overall': 0.35415577626764244, 'recall-overall': 0.23990793201133145, 'f1-measure-overall': 0.2860460206881511}
Test token-level accuracy of NER model: 0.8809.
```
