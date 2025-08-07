source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

function compile_model() {
    mkdir -p build; 
    cd build;

    cmake ..;
    if [ $? -ne 0 ]; then
        echo "ERROR: generate makefile failed!"
        exit 1
    fi

    cmake --build . -j;
    if [ $? -ne 0 ]; then
        echo "ERROR: compile test failed!"
        exit 1
    else
        echo "INFO: compile test succeed!"
    fi
    cd -;
   
}

compile_model
