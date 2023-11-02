from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample.json
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="train a deeplearning model with three layers in pytorch on iris dataset and predict species of this data point={'Sepal.Length':5.1, 'Sepal.Width':3.5, 'Petal.Length':1.4, 'Petal.Width':0.2 }.")
# This initiates an automated chat between the two agents to solve the task