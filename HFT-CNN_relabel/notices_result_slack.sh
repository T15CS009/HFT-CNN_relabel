#!/bin/sh

set -eu

#################### 設定 ####################
### Incoming WebHooksのURL
# WEBHOOKURL="https://hooks.slack.com/services/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"
WEBHOOKURL="https://hooks.slack.com/services/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"

### slack 送信チャンネル
# CHANNEL=${CHANNEL:-"#scripts"}

### slack 送信名
BOTNAME=${BOTNAME:-"titan-v"}
# BOTNAME=${BOTNAME:-"${HOSTNAME}"}

### slack アイコン
ICON='icon_url' # 'icon_url'か'icon_emoji'を選択可能
FACEICON=${FACEICON:-":ghost:"}   # slack上のアイコンから選択可能

### 見出しとなるようなメッセージ
MESSAGE=${MESSAGE:-""}

#################### 設定 ####################


#メッセージを保存する一時ファイル
MESSAGEFILE=$(mktemp -t webhooks.XXXXXX)
trap "
rm ${MESSAGEFILE}
" 0

usage_exit() {
    echo "Usage: $0 [-m message] [-c channel] [-i icon] [-n botname]" 1>&2
    exit 0
}

while getopts c:i:n:m: opts
do
    case $opts in
        c)
            CHANNEL=$OPTARG
            ;;
        i)
            FACEICON=$OPTARG
            ;;
        n)
            BOTNAME=$OPTARG
            ;;
        m)
            MESSAGE=$OPTARG"\n"
            ;;
        \?)
            usage_exit
            ;;
    esac
done

if [ -p /dev/stdin ] ; then
    #改行コードをslack用に変換
    cat - | tr '\n' '\\' | sed 's/\\/\\n/g'  > ${MESSAGEFILE}
else
    echo "nothing stdin"
    exit 1
fi

# WEBMESSAGE='```'`cat ${MESSAGEFILE}`'```'
WEBMESSAGE=`cat ${MESSAGEFILE}`

# Incoming WebHooks送信
# curl -s -S -X POST --data-urlencode "payload={\"channel\": \"${CHANNEL}\", \"username\": \"${BOTNAME}\", \"${ICON}\": \"${FACEICON}\", \"text\": \"${MESSAGE}${WEBMESSAGE}\" }" ${WEBHOOKURL} >/dev/null
curl -s -S -X POST --data-urlencode "payload={\"username\": \"${BOTNAME}\", \"${ICON}\": \"${FACEICON}\", \"text\": \"A script has exited.\n \`\`\`\n${MESSAGE}${WEBMESSAGE}\`\`\` \" }" ${WEBHOOKURL} >/dev/null
