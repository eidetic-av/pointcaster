#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "graph_types.h"
#include "../logger.h"

namespace pc::graph {

	class OperatorGraph {
	public:
		OperatorGraph(std::string_view id);
		~OperatorGraph();

		void draw();

		void add_input_node(NodeDataVariant node_data);
		void add_output_node(NodeDataVariant node_data);

		void add_link(Link new_link);
		void remove_link(int detached_link_id);

		std::vector<int> path_to(int target_node_id) const;

		template<typename T>
		int node_id_from_reference(const T& node_data_reference_variant)
		{
			for (const auto& node : _nodes) {
				if (std::holds_alternative<std::reference_wrapper<T>>(node.node_data)) {
					const auto& node_config = std::get<std::reference_wrapper<T>>(node.node_data).get();
					if (&node_config == &node_data_reference_variant) {
						return node.id;
					}
                }
			}
			return -1;
		}

	private:
		std::string _id;
		std::vector<Node> _nodes;
		std::vector<Link> _links;
	};

} // namespace pc::graph
