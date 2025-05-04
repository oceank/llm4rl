# llm4rl
Utilize the common knowledge captured in a pretrained LLM to improve the learning efficiency of RL

## Anaconda Environment Setup
Use the yaml file, llmrl.yml, to create the anaconda environment

## Three Groups of Scripts
### LLM Agent Evaluation
- evaluate_LLM_agent.py
- slurm_job_llm_agent_evaluation.bash
- submit_jobs_llm_agent_evaluation.bash
### Trajectory Collection By LLM Agent
- trajs_collection_using_llm.py
- slurm_job_trajs_collection_using_llm.bash
- submit_jobs_trajs_collection_using_llm.bash
### DQN Agent - Training and Evaluation
- dqn.py (3-D state tensor by onehot encoding), dqn_2D.py (2-D state tensor with tile value)
- run_dqn.py
- slurm_job_dqn.bash
- submit_jobs_dqn.bash
### Others
- utility.py: method and class definitions for LLM agent
- toy_agent.py: script for evaluating an LLM agent

## Notes for using the ChatGPT API
- Scripts that have an option to use the ChatGPT API: trajs_collection_using_llm.py and toy_agent.py
- When it is going to apply ChatGPT API, use the option, "--use_chatgpt" together with the option, "--llm_model_name", for specifying the intended GPT model
- It is assumed that the API key of the ChatGPT API is saved to the environment variable, "OPENAI_API_KEY", in the file, .env, that locates at the root of this repo.
