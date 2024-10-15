docker network create opensearch-net

docker run -e OPENSEARCH_JAVA_OPTS="-Xms512m -Xmx512m" -e discovery.type="single-node" \
  -e DISABLE_SECURITY_PLUGIN="true" -e bootstrap.memory_lock="true" \
  -e cluster.name="opensearch-cluster" -e node.name="os01" \
  -e plugins.neural_search.hybrid_search_disabled="true" \
  -e DISABLE_INSTALL_DEMO_CONFIG="true" \
  --ulimit nofile="65536:65536" --ulimit memlock="-1:-1" \
  --net opensearch-net --restart=no \
  -v opensearch-data:/usr/share/opensearch/data \
  -p 9200:9200 \
  --name=opensearch-single-node \
  opensearchproject/opensearch:latest