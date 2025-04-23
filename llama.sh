#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 python ./main.py serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8100 \
    --max-model-len 14000 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=1 python ./main.py serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8200 \
    --max-model-len 14000 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens 
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM 
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will 
# introduce "vllm connect" to connect between prefill and decode instances
# python3 disagg_prefill_proxy_server.py &
# sleep 1


# run until Ctrl+C
while true; do
  sleep 1
done

curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": " reat base for exploring the nearby islands, including Aegina, Poros, and Hydra. Take a ferry or a boat to explore these beautiful islands and enjoy the stunni",
"max_tokens": 1000,
"temperature": 0
}'

curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": " reat base for exploring the nearby islands, including Aegina, Poros, and Hydra. Take a ferry or a boat to explore these beautiful islands and enjoy the stunning scenery.\n7. Visit the Monastiraki Flea Market: The Monastiraki Flea Market is a great place to find unique souvenirs and antiques. It's held every Sunday, and it's a great place to experience the local culture.\n8. Take a stroll through the National Garden: The National Garden is a beautiful park in the heart of the city, and it's a great place to take a stroll and enjoy the scenery. It's also home to a variety of plants and flowers, including a beautiful rose garden.\n9. Visit the Temple of Olympian Zeus: The Temple of Olympian Zeus is a ancient temple that was built in the 2nd century BC. It's a great place to learn about the history of ancient Greece and to see some of the city's most impressive architecture.\n10. Enjoy the street food: Athens has a lively street food scene, with everything from souvlaki to Greek pastries. Be sure to try some of the local street food, including the famous Greek gyro.\n\nOverall, Athens is a city that has something for everyone, from history and culture to nightlife and cuisine. With these tips, you'll be well on your way to experiencing all that Athens has to offer.\n\n**Best Time to Visit Athens**\nThe best time to visit Athens is from September to November or from March to May, when the weather is mild and pleasant. These periods offer the best combination of comfortable temperatures and fewer tourists.\n\n**Getting Around Athens**\nAthens has a well-developed public transportation system, including buses, metro lines, and trams. You can also take taxis or ride-sharing services to get around the city.\n\n**Safety in Athens**\nAthens is generally a safe city, but as w",
"max_tokens": 5000,
"temperature": 0
}'
