#include <optional>

namespace pc::types {

	template<typename T>
	using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

	template<typename T>
	struct is_optional : std::false_type {};
	template<typename T>
	struct is_optional<std::optional<T>> : std::true_type {};
	template<typename T>
	inline constexpr bool is_optional_v = is_optional<remove_cvref_t<T>>::value;

}