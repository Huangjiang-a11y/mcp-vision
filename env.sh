# 给 shell 用：把 env.local 里的 KEY=VALUE 导出（set -a 让赋值自动 export）
_d=$(cd "$(dirname "$0")" 2>/dev/null && pwd)
[ -f "$_d/env.local" ] && set -a && . "$_d/env.local" && set +a
unset _d
