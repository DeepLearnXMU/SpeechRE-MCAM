#! /bin/bash
export Prefix= ...
export IWSLT_ROOT=$Prefix/IWSLT
export FAIRSEQ_ROOT=$Prefix/fairseq
export WAV2VEC_ROOT=$IWSLT_ROOT/Pre-trained_models/Wav2Vec
export WAV2VEC_BASE=$IWSLT_ROOT/Pre-trained_models/Wav2Vec-base
export BART_ROOT=$IWSLT_ROOT/Pre-trained_models/BART-large
export BART_BASE=$IWSLT_ROOT/Pre-trained_models/BART-base
export REBEL_ROOT=$IWSLT_ROOT/Pre-trained_models/REBEL-large
export DATA_ROOT=$IWSLT_ROOT/Data
export SAVE_DIR=$IWSLT_ROOT/Save
export HYDRA_FULL_ERROR=1

export CUDA_HOME=/opt/conda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib

/opt/conda/bin/python $FAIRSEQ_ROOT/fairseq_cli/hydra_train.py \
    --config-dir $IWSLT_ROOT/config/ \
    --config-name speechre_tacred_part_part.yaml \
    | tee $IWSLT_ROOT/logs/10990.train.log 2>&1