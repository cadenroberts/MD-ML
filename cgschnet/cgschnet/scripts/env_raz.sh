#!/bin/bash

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_EXE="/u/kbachelor/.local/bin/micromamba";
export MAMBA_ROOT_PREFIX="/u/kbachelor/y";
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    if [ -f "/u/kbachelor/y/etc/profile.d/micromamba.sh" ]; then
        . "/u/kbachelor/y/etc/profile.d/micromamba.sh"
    else
        export  PATH="/u/kbachelor/y/envs/default/bin:$PATH"  # extra space after export prevents interference from conda init
    fi
fi
unset __mamba_setup
# <<< mamba initialize <<<


