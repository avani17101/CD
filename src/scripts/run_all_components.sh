model_type='colormnist'
dset='mnist'
pairs_val=8
bottleneck_name="conv2" ##same for all mnist
##Step 0.0: Saving class proto-types
python utils/proto_save.py --model_type $model_type --dset $dset --pairs_vals $pairs_val --bottleneck_name $bottleneck_name

## Step 1: Mapping Module learning
## Step 1.1: save student outputs and dino activations for given concept sets
python save_student_outs.py --model_type $model_type --dset $dset --pairs_vals $pairs_val --bottleneck_name $bottleneck_name
python save_dino_outs.py --model_type $model_type --pairs_vals $pairs_val

## Step 1.2: Learn mapping module to map teacher to student for concept sets

python map_teacher_student.py --model_type $model_type --dset $dset --pairs_vals $pairs_val
## Step 2: Learning CAVs in Mapped Teacher Space
python learn_cavs.py --model_type $model_type --dset $dset --pairs_vals $pairs_val
## Step 3: Concept Learning in Student
python CD.py --model_type $model_type --dset $dset --pairs_vals $pairs_val --bottleneck_name $bottleneck_name  ##for distill
# python CD_direct_student.py --model_type $model_type --dset $dset --pairs_vals $pairs_val --bottleneck_name $bottleneck_name ## for no distill but direct concept loss in student