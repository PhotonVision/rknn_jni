/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef RKNN_JAVA_UTILS_FILE_UTILS_H_
#define RKNN_JAVA_UTILS_FILE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read data from file
 *
 * @param path [in] File path
 * @param out_data [out] Read data
 * @return int -1: error; > 0: Read data size
 */
int read_data_from_file(const char *path, char **out_data);

/**
 * @brief Write data to file
 *
 * @param path [in] File path
 * @param data [in] Write data
 * @param size [in] Write data size
 * @return int 0: success; -1: error
 */
int write_data_to_file(const char *path, const char *data, unsigned int size);

/**
 * @brief Read all lines from text file
 *
 * @param path [in] File path
 * @param line_count [out] File line count
 * @return char** String array of all lines, remeber call free_lines() to
 * release after used
 */
char **read_lines_from_file(const char *path, int *line_count);

/**
 * @brief Free lines string array
 *
 * @param lines [in] String array
 * @param line_count [in] Line count
 */
void free_lines(char **lines, int line_count);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // RKNN_JAVA_UTILS_FILE_UTILS_H_
