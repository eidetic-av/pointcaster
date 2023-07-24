#pragma once

#include <filesystem>
#include <vector>

namespace path {

/**
 * @brief Get the system path of the pointcaster executable.
 *
 * @return std::filesystem::path Path of the currently running executable.
 */
std::filesystem::path exe_path();

/**
 * @brief Get the system path of the pointcaster directory.
 *
 * @return std::filesystem::path Path of the directory containing the
 * executable.
 */
std::filesystem::path exe_directory();

/**
 * @brief Get the path of the 'data' directory, located next to the pointcaster
 * executable.
 *
 * @return std::filesystem::path Path of the 'data' directory.
 */
std::filesystem::path data_directory();

/**
 * @brief Get the path of the 'data' directory, and create it if it doesn't
 * already exist. executable.
 *
 * @return std::filesystem::path Path of the 'data' directory.
 */
std::filesystem::path get_or_create_data_directory();

/**
 * @brief Loads a file into a vector of bytes.
 *
 * This function takes a file path as input and returns its contents as a vector
 * of bytes (uint8_t).
 *
 * @param file_path A std::filesystem::path object representing the path of the
 * file to be loaded.
 * @return std::vector<uint8_t> A vector containing the bytes of the file.
 */
std::vector<uint8_t> load_file(std::filesystem::path file_path);

/**
 * @brief Saves a vector of bytes into a file.
 *
 * This function takes a vector of bytes (uint8_t) as input and saves its
 * content into a file. If the file already exists, it will be overwritten.
 *
 * @param file_path A std::filesystem_path object of the save file location.
 * @param buffer A std::vector<uint8_t> object containing the bytes to be saved
 * into the file.
 * @return bool Success or failure
 */
bool save_file(std::filesystem::path file_path, std::vector<uint8_t> buffer);

} // namespace path
