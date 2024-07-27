source /home/tanty04/anaconda3/bin/activate cs285
python run_hw2.py --env_name LunarLanderContinuous-v2 \
                                --ep_len 500 \
                                --discount 0.99 \
                                -n 100 \
                                -l 2 \
                                -s 64 \
                                -b 10000 \
                                -lr 0.0001 \
                                --reward_to_go \
                                --exp_name q3_b40000_r0.0001_no_bsl_lunar