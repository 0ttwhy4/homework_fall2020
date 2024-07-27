source /home/tanty04/anaconda3/bin/activate cs285
python run_hw2.py --env_name CartPole-v0 \
                  -n 100 \
                  -b 500 \
                  --nn_baseline \
                  -rtg \
                  --exp_name q1_sb_rtg_ref