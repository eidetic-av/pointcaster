#include <string>
#include "../operators/operator_host_config.gen.h"
#include "../devices/device_config.gen.h"
#include "../camera/camera_config.gen.h"

namespace pc::graph {

	enum class LinkDirection { In, Out, InOut };

	using NodeDataVariant = std::variant<
		operators::OperatorConfigurationVariant,
		 std::reference_wrapper<pc::devices::DeviceConfiguration>,
		 std::reference_wrapper<pc::camera::CameraConfiguration>
	>;

	struct Node {
		int id;
		LinkDirection link_direction;
		NodeDataVariant node_data;
	};

	struct Link {
		int id;
		int output_id;
		int input_id;
	};

	template <typename T>
	struct is_reference_wrapper : std::false_type {};

	template <typename U>
	struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

}