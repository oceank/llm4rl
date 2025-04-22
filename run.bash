#!/bin/bash

llm_model_name="TheBloke/deepseek-llm-7b-chat-GPTQ" #"deepseek-ai/deepseek-llm-7b-chat"
map_names=(3x3) #(2x2OneStep 2x2 3x3 4x4) #8x8)
#map_name=2x2 #"4x4"

tr_eps=15000 #5000 #15000 # number of episodes for training
lr=0.9 # learning rate of Q-learning before epsilon decays to 0
lr_final=0.0001 # learning rate of Q-learning after epsilon decays to 0
ev_eps=25 #25 #25 #5 #25 #100 # number of episodes per evaluation
ev_interval=25 #1 25 #100 # evaluation frequency: e.g., evaluate the policy during training every 100 episodes 
start_epsilon=1.0 # the initial value of the epsilon
final_epsilon=0.0 # the final value of the ep
epsilon_decay=0.0004 #0.0001 # the decayed epsilon value every episode (or step). With the current setting, epsilon will decay to 0 after 10000 episodes

# --is_slippery
# --full_map_observable
full_map_desc_type=2 # 0, 1, 2
history_window_size=0 # 0, 5
tr_config_prefix=" --history_window_size ${history_window_size} --full_map_observable --full_map_desc_type ${full_map_desc_type} --n_episodes_eval ${ev_eps} --n_episodes ${tr_eps} --eval_interval ${ev_interval} "
rl_config=" --learning_rate ${lr} --final_learning_rate ${lr_final} --start_epsilon ${start_epsilon} --final_epsilon ${final_epsilon} --epsilon_decay ${epsilon_decay}"

seeds=(1 2 3 4 5)
#seeds=(1) # 2 3 4 5)
for map_name in "${map_names[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "=== Run experiments with the map ${map_name} seed $seed ==="
        #echo "===> evaluate the random agent for the seed ${seed}"
        #python corral_toy_localLLM.py --update_type random  --seed ${seed} --map_name ${map_name} ${tr_config_prefix}

        echo "===> evaluate the LLM agent for the seed ${seed}"
        cmd="python corral_toy_localLLM.py --update_type llm  --seed ${seed} --map_name ${map_name} ${tr_config_prefix}"
        #python corral_toy_chatgpt_v2.py --update_type llm  --seed ${seed} --map_name ${map_name} ${tr_config_prefix}
        echo -e "CMD: ${cmd}"
        eval ${cmd}


        #echo "===> evaluate the RL agent for the seed ${seed}" # Q-learning
        #python corral_toy_localLLM.py --update_type rl  --seed ${seed} --map_name ${map_name} ${tr_config_prefix} ${rl_config}


    done
done

#python corral_toy_localLLM.py --update_type random --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 1

#python corral_toy_localLLM.py --update_type random --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 2
#python corral_toy_localLLM.py --update_type random --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 3
#python corral_toy_localLLM.py --update_type random --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 4
#python corral_toy_localLLM.py --update_type random --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 5

#python corral_toy_localLLM.py --update_type llm --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 1
#python corral_toy_localLLM.py --update_type llm --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 2
#python corral_toy_localLLM.py --update_type llm --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 3
#python corral_toy_localLLM.py --update_type llm --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 4
#python corral_toy_localLLM.py --update_type llm --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 5

#python corral_toy_localLLM.py --update_type rl --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 1
#python corral_toy_localLLM.py --update_type rl --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 2
#python corral_toy_localLLM.py --update_type rl --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 3
#python corral_toy_localLLM.py --update_type rl --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 4
#python corral_toy_localLLM.py --update_type rl --n_episodes_eval 100 --n_episodes 20000 --eval_interval 100 --seed 5

