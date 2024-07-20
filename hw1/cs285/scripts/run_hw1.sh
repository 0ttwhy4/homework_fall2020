python3 run_hw1.py --expert_policy_file /home/tanty04/GitRepo/homework_fall2020/hw1/cs285/policies/experts/Humanoid.pkl \
                   --env_name Humanoid-v2 \
                   --exp_name bc_human \
                   --n_iter 5 \
                   --do_dagger \
                   --expert_data /home/tanty04/GitRepo/homework_fall2020/hw1/cs285/expert_data/expert_data_Humanoid-v2.pkl \
                   --n_layers 3 \
                   --train_batch_size 500 \
                   # --video_log_freq -1 \