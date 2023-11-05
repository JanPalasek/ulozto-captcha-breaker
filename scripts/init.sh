if ! grep -q "# DEVCONTAINER INIT" ~/.bashrc; then
    echo "Initializing..."
    echo "# DEVCONTAINER INIT #" >> ~/.bashrc;
    echo "export USER_NAME=$(id -un)" >> ~/.bashrc;
    echo "export USER_ID=$(id -u)" >> ~/.bashrc;
    echo "export USER_GID=$(id -g)" >> ~/.bashrc;
    echo "export USER_GNAME=$(id -gn)" >> ~/.bashrc;
    echo "export DOCKER_GID=$(getent group docker | cut -d: -f3)" >> ~/.bashrc;
fi
echo "Initialization complete..."