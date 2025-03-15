#!/bin/bash

API_URL="http://localhost:3000/chat"

CONCURRENT_REQUESTS=5

make_request() {
    local id=$1
    echo "$id： "
    time curl -s -X POST $API_URL \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"Hi (#$id)\"}"
}

export -f make_request
export API_URL

echo "并发测试： 同时发送 $CONCURRENT_REQUESTS 个请求"
echo "-------------------"

seq 1 $CONCURRENT_REQUESTS | parallel -j$CONCURRENT_REQUESTS make_request

echo "-------------------"
echo "OK" 