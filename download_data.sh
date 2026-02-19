lps=(en-de en-fr en-nl en-pt en-ko)

for lp in "${lps[@]}"; do
    wget https://raw.githubusercontent.com/WMT-Chat-task/chat-task-2024-data/main/test/${lp}.csv -O downloaded_data/wmt24_chat_test_blind.${lp}.csv
done