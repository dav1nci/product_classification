STR_TO_RUN="
docker run -dt --name llm \
	-v $PWD/notebooks/:/notebooks/ \
	-v $PWD/data/:/data/ \
	-v $PWD/src/:/src/ \

	-p 127.0.0.1:8108:8888 \

	--gpus '\"device=0\"' \
	--ipc=host \
        --network services_ai_network \
        llm:huggingface-gpu
"

# jupyter port forwarding ssh -L 127.0.0.1:8108:127.0.0.1:8108 -L 127.0.0.1:6106:127.0.0.1:6106 deeper

docker container stop llm && docker container rm llm
eval $STR_TO_RUN
