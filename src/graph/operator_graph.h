#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "graph_types.h"

namespace pc::graph {
	class OperatorGraph {
	public:

		OperatorGraph(std::string_view id);
		~OperatorGraph();

		void add_input_node(NodeDataVariant node_data);
		void add_output_node(NodeDataVariant node_data);

		void remove_node(const std::string& id);

		void draw();

	private:

		std::string _id;
		std::vector<Node> _nodes;
		std::vector<Link> _links;

	};
}