#!/bin/bash

# Create the custom hooks

cd .git/hooks/
if [ $? -ne 0 ]; then
    exit 1
fi

ln -s ../../hooks/pre-push pre-push
ln -s ../../hooks/prepare-commit-msg prepare-commit-msg

