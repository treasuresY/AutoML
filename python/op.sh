#!/usr/bin/env bash

set -x
op=$1

# 获取当前脚本所在目录
DIR_PATH="$(cd "$(dirname "$0")"; pwd -P)"

# 镜像信息
IMAGE_FULL="registry.cn-hangzhou.aliyuncs.com/treasures/automl:latest"
IMAGE_NAME="registry.cn-hangzhou.aliyuncs.com/treasures/automl"
IMAGE_TAG="latest"

DOCKERFILE="${DIR_PATH}/automl.Dockerfile"
CONTEXT="${DIR_PATH}"

DEPLOYMENT_YAML="${DIR_PATH}/automl-deployment.yaml"
PV_PVC_YAML="${DIR_PATH}/automl-pv-pvc.yaml"


is_file_exist() {
    # 此方法用于检验文件是否存在，若不存在，返回1; 存在，正常退出，返回0.
    # params:
    #   file_path: 文件路径
    if [ -z "$1" ]; then
        echo "need a param {file_path}"
        return 1
    fi

    file_path=$1
    if [ ! -f "${file_path}" ]; then
        echo "${file_path} does not exist."
        return 1
    fi
}

is_dir_exist() {
    # 此方法用于检验目录是否存在，若不存在，返回1; 存在，正常退出，返回0.
    # params:
    #   dir_path: 目录路径
    if [ -z "$1" ]; then
        echo "need a param {dir_path}"
        return 1
    fi

    dir_path=$1
    if [ ! -d "${dir_path}" ]; then
        echo "${dir_path} does not exist"
        return 1
    fi
}

is_image_exist() {
    # 此方法用于检验镜像是否存在，若不存在，返回1; 存在，正常退出，返回0.
    # params:
    #   image_full: 镜像全称
    if [ -z "$1" ]; then
        echo "need a param {image_full}"
        return 1
    fi

    image_full=$1
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q ${image_full}
    if [ $? -ne 0 ]; then
        echo "${image_full} dose not exist"
        return 1
    fi
}

op_yaml() {
    # 此方法用于操作.yaml后缀文件, 需要 2 个参数。
    # params:
    #   op_type: 操作类型。例如apply、delete等
    #   yaml_path: yaml文件路径。
    if [ -z "$1" ]; then
        echo "need a param {op_type}"
        return 1
    fi

    if [ -z "$2" ]; then
        echo "need a param {yaml_path}"
        return 1
    fi

    op_type=$1
    yaml_path=$2

    is_file_exist ${yaml_path}
    if [ $? -ne 0 ]; then
        return 1
    fi

    case ${op_type} in
        apply)
            echo "applying ${yaml_path}"
            kubectl apply -f ${yaml_path}
            if [ $? -ne 0 ]; then
                echo "fail to apply ${yaml_path}"
                return 1
            fi
            echo "apply ${yaml_path} succseefully!"
            ;;
        delete)
            echo "deleting ${yaml_path}"
            kubectl delete -f ${yaml_path}
            if [ $? -ne 0 ]; then
                echo "fail to delete ${yaml_path}"
                return 1
            fi
            echo "delete ${yaml_path} successfully!"
            ;;
        *)  # 后续扩展operations
            echo "${op_type} is not support"
            ;;
    esac
}

op_iamge() {
    # 此方法用于操作docker镜像，需要四个参数。
    # params:
    #   op_type: 操作类型。例如build、rmi等;
    #   image_full: 镜像全称（镜像name：镜像tag）;
    #   dokerfile_path: dockerfile路径;
    #   context_path: docker build命令构建上下文路径。
    if [ -z "$1" ]; then
        echo "need a param {op_type}"
        return 1
    fi

    if [ -z "$2" ]; then
        echo "need a param {image_full}"
        return 1
    fi

    context_path="."
    if [ ! -z "$4" ]; then
        context_path=$4
    fi

    op_type=$1
    image_full=$2
    # 从字符串的结束处删除最短匹配的子串, 此处删除镜像tag。例如image:tag -> image
    # image_name=${image_full%:*}

    case ${op_type} in
        build)
            is_image_exist ${image_full}
            if [ $? -eq 0 ]; then
                return 1
                # 镜像存在，可以考虑删除原镜像
                # docker rmi ${image_full}
            fi

            if [ -z "$3" ]; then
                echo "need a param {dokerfile_path}"
                return 1
            fi
            dockerfile_path=$3
            is_file_exist ${dockerfile_path}
            if [ $? -ne 0 ]; then
                return 1
            fi

            echo "building ${image_full}"
            docker build -t "${image_full}" -f "${dockerfile_path}" "${context_path}"
            if [ $? -ne 0 ]; then
                echo "faid to build ${image_full}"
                return 1
            fi
            echo "build successfully!"
            ;;
        rmi)
            echo "deleting ${image_full}"
            is_image_exist ${image_full}
            if [ $? -ne 0 ]; then
                return 1
            fi
            docker rmi ${image_full}
            if [ $? -ne 0 ]; then
                echo "fail to rmi ${image_full}"
                return 1
            fi
            echo "delete ${image_full} successfully!"
            ;;
        *)  # 后续扩展operations
            echo "${op_type} is not support"
            ;;
    esac
}


# 上线automl
up_automl() {
    op_iamge build ${IMAGE_FULL} ${DOCKERFILE} ${CONTEXT}
    # apply automl-pv-pvc.yaml
    op_yaml apply ${PV_PVC_YAML}
    # apply automl-deployment.yaml
    op_yaml apply ${DEPLOYMENT_YAML}
    
}

down_automl() {
    # delete automl-deployment.yaml
    op_yaml delete ${DEPLOYMENT_YAML}
    sleep 4
    op_yaml delete ${PV_PVC_YAML}
    op_iamge rmi ${IMAGE_FULL}
}


case "$op" in
up)
    echo "Deploying automl server"
    up_automl
    ;;
down)
    echo "Closing automl server"
    down_automl
    ;;
*)
    echo "${op} does not support"
    ;;
esac
