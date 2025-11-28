if not contains "$HOME/Desktop/targeted_MLCE_optimization/artemis-tools-3.1.1/.uv" $PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "$HOME/Desktop/targeted_MLCE_optimization/artemis-tools-3.1.1/.uv" $PATH
end
