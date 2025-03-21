#!/bin/bash


API_CHAT_URL="http://localhost:3000/chat"
API_ROLLBACK_URL="http://localhost:3000/rollback"
API_DELETE_URL="http://localhost:3000/delete_session"
SESSION_ID="test-session-$(date +%s)"  


GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

print_step() {
    echo -e "${BLUE}[STEP] $1${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}


send_chat_message() {
    local message=$1
    print_step "发送消息: \"$message\""
    
    response=$(curl -s -X POST $API_CHAT_URL \
         -H "Content-Type: application/json" \
         -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"$message\"}")
    

    model_response=$(echo $response | sed 's/.*"response":"\(.*\)"}/\1/' | sed 's/\\n/\n/g')
    
    print_info "模型回复:"
    echo -e "$model_response"
    echo "-------------------"
}


do_rollback() {
    local target_turn=$1
    
    if [ -z "$target_turn" ]; then
        print_step "执行回滚到上一个快照"
        rollback_data="{\"session_id\":\"$SESSION_ID\"}"
    else
        print_step "执行回滚到第 $target_turn 轮"
        rollback_data="{\"session_id\":\"$SESSION_ID\",\"rollback_to_turn\":$target_turn}"
    fi
    
    rollback_response=$(curl -s -X POST $API_ROLLBACK_URL \
         -H "Content-Type: application/json" \
         -d "$rollback_data")
    

    success=$(echo $rollback_response | grep -o '"success":true' | wc -l)
    message=$(echo $rollback_response | sed 's/.*"message":"\([^"]*\)".*/\1/')
    current_turn=$(echo $rollback_response | sed 's/.*"current_turn":\([0-9]*\).*/\1/')
    
    if [ "$success" -eq 1 ]; then
        print_info "回滚成功: $message"
        print_info "当前轮次: $current_turn"
    else
        print_error "回滚失败: $message"
        print_info "当前轮次: $current_turn"
    fi
    
    echo "-------------------"
}

delete_session() {
    local session_id=$1
    print_step "删除会话: $session_id"
    
    delete_response=$(curl -s -X POST $API_DELETE_URL \
         -H "Content-Type: application/json" \
         -d "{\"session_id\":\"$session_id\"}")
    
    success=$(echo $delete_response | grep -o '"success":true' | wc -l)
    message=$(echo $delete_response | sed 's/.*"message":"\([^"]*\)".*/\1/')
    
    if [ "$success" -eq 1 ]; then
        print_info "删除成功: $message"
    else
        print_error "删除失败: $message"
    fi
    
    echo "-------------------"
}

print_info "开始测试基本对话和删除功能"
print_info "会话 ID: $SESSION_ID"
echo "===================="

send_chat_message "Hello, please introduce yourself"
send_chat_message "Let's move to Paris. I like France."
send_chat_message "What can you do?"

# 测试回滚功能
print_info "测试回滚到第1轮"
do_rollback 2

# 验证回滚后的状态
send_chat_message "What can you do?"

# 测试删除存在的会话
print_info "测试删除存在的会话"
delete_session "$SESSION_ID"

# 测试删除已经删除的会话
print_info "测试删除已经删除的会话"
delete_session "$SESSION_ID"

# 测试删除一个从未存在的会话
print_info "测试删除一个从未存在的会话"
delete_session "non-existent-session-id"

# 验证会话已被删除（尝试发送消息）
print_info "验证会话已被删除（尝试发送新消息）"
send_chat_message "What can you do?"

echo "===================="
print_info "测试完成"



# curl -s -X POST http://localhost:3000/rollback \
#      -H "Content-Type: application/json" \
#      -d '{"session_id":"session1","rollback_to_turn":2}'
