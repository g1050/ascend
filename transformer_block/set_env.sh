if [ "$1" = "debug" ]; then
    export ASDOPS_LOG_TO_STDOUT=1
    export ASDOPS_LOG_LEVEL=DEBUG
else
    export ASDOPS_LOG_TO_STDOUT=0
    export ASDOPS_LOG_LEVEL=ERROR
fi