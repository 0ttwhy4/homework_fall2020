source /home/tanty04/anaconda3/bin/activate cs285
python run_hw2.py --env_name InvertedPendulum-v2 \
                                --ep_len 1000 \
                                --discount 0.9 \
                                -n 10 \
                                -l 2 \
                                -s 64 \
                                -b 50 \
                                -lr 1e-3 \
                                -rtg \
                                --exp_name q2_b500_r1e-3_n3