# Chem-Agent
Retrievel-Augmented Generation (a.k.a RAG) agent in KCLab.

## Run elasticsearch

```shell-session
% sudo docker network create elastic

% sudo docker run -it --name es01 --net elastic -p 9200:9200 -e=xpack.security.enabled=false  -e="ES_JAVA_OPTS=-Xms1g -Xmx1g" -e=discovery.type=single-node -m 2GB elasticsearch:8.12.0
```

If elasticsearch exited with code 137, it is sent a `SIGKILL`, most possibly OOM-killed. The solution is increase mem allocated with `-m` arg.

If elasticsearch exited with code 78:

```shell-session
% sudo sysctl -w vm.max_map_count=524288
```

If `xpack.security` is `True`, Use following command to reset passwds:

```shellsession
% docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
% docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-create-enrollment-token -s kibana
```
