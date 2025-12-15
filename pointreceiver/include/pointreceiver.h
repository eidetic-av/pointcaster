/**
 * @file pointreceiver.h
 * @brief Public API for the PointReceiver library.
 *
 * This header provides a C API for the PointReceiver library,
 * which handles parameter updates and point cloud streams from a
 * local or remote Pointcaster instance.
 */

#pragma once
#ifndef POINTRECEIVER_H
#define POINTRECEIVER_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __ANDROID__
#include <jni.h>
#define PR_EXPORT JNIEXPORT
#define APPNAME "PointReceiver"
#elif _WIN32
#define PR_EXPORT __declspec(dllexport)
#else
#define PR_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque structure representing the PointReceiver context.
 *
 * The internal details of this structure are hidden from API users.
 */
typedef struct pointreceiver_context pointreceiver_context;

/**
 * @brief Enumerates the types of messages that can be received.
 */
typedef enum {
  POINTRECEIVER_MSG_TYPE_CONNECTED = 0,           /**< Connected message */
  POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT,          /**< Client heartbeat */
  POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT_RESPONSE, /**< Response to client heartbeat */
  POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE,          /**< Parameter update message */
  POINTRECEIVER_MSG_TYPE_PARAMETER_REQUEST,         /**< Parameter request message */
  POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE,           /**< Pointcaster server endpoint update message */
  POINTRECEIVER_MSG_TYPE_UNKNOWN                    /**< Unknown message type */
} pointreceiver_message_type;

/**
 * @brief Enumerates the possible types for parameter values.
 */
typedef enum {
  POINTRECEIVER_PARAM_VALUE_FLOAT = 0,  /**< Float value */
  POINTRECEIVER_PARAM_VALUE_INT,        /**< Integer value */
  POINTRECEIVER_PARAM_VALUE_FLOAT3,     /**< 3D float vector */
  POINTRECEIVER_PARAM_VALUE_FLOAT4,     /**< 4D float vector */
  POINTRECEIVER_PARAM_VALUE_FLOAT3LIST, /**< List of 3D float vectors */
  POINTRECEIVER_PARAM_VALUE_FLOAT4LIST, /**< List of 4D float vectors */
  POINTRECEIVER_PARAM_VALUE_AABBLIST,   /**< List of Axis-aligned bounding boxes
                                         */
  POINTRECEIVER_PARAM_VALUE_CONTOURSLIST, /**< List of contours. 2D polygons
                                             stored as a list of vertex
                                             locations in pointcaster camera
                                             space */
  POINTRECEIVER_PARAM_VALUE_UNKNOWN       /**< Unknown parameter type */
} pointreceiver_param_value_type;

/**
 * @brief Structure representing a 2D float vector.
 *
 * Contains the x and y components.
 */
typedef struct {
  float x; /**< X component of the 3D vector */
  float y; /**< Y component of the 3D vector */
} pointreceiver_float2_t;

/**
 * @brief Structure representing a 3D float vector.
 *
 * Contains the x, y, and z components.
 */
typedef struct {
  float x; /**< X component of the 3D vector */
  float y; /**< Y component of the 3D vector */
  float z; /**< Z component of the 3D vector */
} pointreceiver_float3_t;

/**
 * @brief Structure representing a 3D float vector.
 *
 * Contains the x, y, and z components.
 */
typedef struct {
  float x; /**< X component of the 4D vector */
  float y; /**< Y component of the 4D vector */
  float z; /**< Z component of the 4D vector */
  float w; /**< W component of the 4D vector */
} pointreceiver_float4_t;

/**
 * @brief Structure representing an Axis-Aligned Bounding Box (AABB).
 *
 * This structure contains the minimum and maximum coordinates of the box.
 */
typedef struct {
  float min[3]; /**< Minimum coordinate (x, y, z) */
  float max[3]; /**< Maximum coordinate (x, y, z) */
} pointreceiver_aabb_t;

/**
 * @brief Structure representing a list of AABBs.
 *
 * This structure encapsulates a pointer to an array of AABBs and the number
 * of elements in the list.
 */
typedef struct {
  pointreceiver_aabb_t *data; /**< Pointer to an array of AABB values */
  size_t count; /**< Number of AABB values in the array */
} pointreceiver_aabb_list_t;

/**
 * @brief Structure representing a list of 3D float vectors.
 *
 * This structure encapsulates a pointer to an array of 3D float vector values and
 * the number of elements in the array.
 */
typedef struct {
  pointreceiver_float3_t *data; /**< Pointer to an array of 3D float vector values */
  size_t count;                /**< Number of 3D float vector values in the array */
} pointreceiver_float3_list_t;

/**
 * @brief Structure representing a list of 4D float vectors.
 *
 * This structure encapsulates a pointer to an array of 4D float vector values and
 * the number of elements in the array.
 */
typedef struct {
  pointreceiver_float4_t *data; /**< Pointer to an array of 3D float vector values */
  size_t count;                /**< Number of 3D float vector values in the array */
} pointreceiver_float4_list_t;

typedef struct {
  pointreceiver_float2_t *data; /**< Vertex list*/
  size_t count;                /**< Number of vertices in the countour */
} pointreceiver_contour_t;

typedef struct {
  pointreceiver_contour_t *data; /**< Contours list*/
  size_t count;                /**< Number of contours in the list */
} pointreceiver_contours_list_t;


/**
 * @brief Structure representing an endpoint update from a Pointcaster server.
 *
 * This structure encapsulates a pointer to an endpoint string,
 * the number of chars in the string, and whether or not the endpoint is active.
 */
typedef struct {
  size_t port; /**< Ephemeral port of the endpoint */
  bool active; /**< Whether the endpoint has become active or been disabled in this update */
} pointreceiver_endpoint_update;


/**
 * @brief Structure representing a synchronized message.
 *
 * This structure encapsulates the message type, an identifier, and a union
 * holding the associated value.
 */
typedef struct {
  pointreceiver_message_type message_type; /**< Type of the message */
  char id[256];                            /**< Identifier string */
  pointreceiver_param_value_type
      value_type; /**< Type of the value contained in the union */

  union {
    float float_val;                         /**< Float value */
    int int_val;                             /**< Integer value */
    pointreceiver_float3_t float3_val;       /**< 3D float vector value */
    pointreceiver_float4_t float4_val;       /**< 4D float vector value */
    pointreceiver_float3_list_t
        float3_list_val; /**< List of 3D float vector values */
    pointreceiver_float4_list_t
        float4_list_val; /**< List of 4D float vector values */
    pointreceiver_aabb_list_t aabb_list_val; /**< List of AABB values */
    pointreceiver_contours_list_t contours_list_val; /**< List of contours */
    pointreceiver_endpoint_update endpoint_update_val; /**< Endpoint Update value */
  } value;               /**< Union holding the message value */

} pointreceiver_sync_message;

/**
 * @brief A point cloud frame received from a Pointcaster instance.
 */
typedef struct {
  int point_count; /**< Number of points in the frame */
  void *positions; /**< Pointer to the point positions buffer */
  void *colours;   /**< Pointer to the point colours buffer */
} pointreceiver_pointcloud_frame;

/**
 * @brief Creates a new PointReceiver context.
 *
 * @return A pointer to a new pointreceiver_context, or NULL on failure.
 */
PR_EXPORT pointreceiver_context *pointreceiver_create_context(void);

/**
 * @brief Destroys a PointReceiver context.
 *
 * Frees resources associated with the given context.
 *
 * @param ctx Pointer to the context to be destroyed.
 */
PR_EXPORT void pointreceiver_destroy_context(pointreceiver_context *ctx);

/**
 * @brief Sets the context's client name
 *
 * @param ctx Pointer to the context to target
 * @param client_name Name for this client to send to server on connected
 */
PR_EXPORT void pointreceiver_set_client_name(pointreceiver_context *ctx,
                                             const char *client_name);

/**
 * @brief Starts the message receiver thread.
 *
 * This function initialises the message receiving mechanism and begins
 * listening for messages from a specified pointcaster address.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param pointcaster_address The network address of the pointcaster.
 * @return 0 on success, non-zero on failure.
 */
PR_EXPORT int pointreceiver_start_message_receiver(pointreceiver_context *ctx,
                                     const char *pointcaster_address);

/**
 * @brief Stops the message receiver thread.
 *
 * Stops listening for messages and terminates the associated thread.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @return 0 on success, non-zero on failure.
 */
PR_EXPORT int pointreceiver_stop_message_receiver(pointreceiver_context *ctx);

/**
 * @brief Dequeues a message from the message queue.
 *
 * Attempts to dequeue a message within a specified timeout period. If a message
 * is successfully retrieved, the caller must later call
 * pointreceiver_free_sync_message to release any allocated resources associated
 * with the message.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param[out] out_message Pointer to a structure where the dequeued message
 * will be stored.
 * @param timeout_ms Timeout in milliseconds to wait for a message.
 * @return true if a message was successfully dequeued, false otherwise.
 */
PR_EXPORT bool pointreceiver_dequeue_message(pointreceiver_context *ctx,
                                               pointreceiver_sync_message *out_message,
                                               int timeout_ms);

/**
 * @brief Frees resources associated with a synchronized message.
 *
 * Releases any dynamically allocated memory within the provided
 * pointreceiver_sync_message. This function must be called after the
 * message has been processed to avoid memory leaks.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param[out] out_message Pointer to the synchronized message whose resources
 * will be freed.
 * @return true if the resources were successfully released, false otherwise.
 */
PR_EXPORT bool pointreceiver_free_sync_message(pointreceiver_context *ctx,
                                                 pointreceiver_sync_message *out_message);

/**
 * @brief Starts the point cloud receiver thread.
 *
 * Initialises the receiver for point cloud streams on a separate socket.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param pointcaster_address The network address for the point cloud stream.
 * @return 0 on success, non-zero on failure.
 */
PR_EXPORT int pointreceiver_start_point_receiver(pointreceiver_context *ctx,
                                                 const char *pointcaster_address);

/**
 * @brief Stops the point cloud receiver thread.
 *
 * Terminates the point cloud stream receiver.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @return 0 on success, non-zero on failure.
 */
PR_EXPORT int pointreceiver_stop_point_receiver(pointreceiver_context *ctx);

/**
 * @brief Dequeues a point cloud frame from pointcaster's live frame queue if
 * one is available.
 *
 * This function attempts to dequeue a complete point cloud frame and fills in
 * the provided frame structure with the point count and buffer pointers.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param[out] out_frame Pointer to a pointreceiver_pointcloud_frame structure
 * where the dequeued frame information will be stored.
 * @param timeout_ms Timeout in milliseconds to wait for a point cloud frame.
 * @return true if a frame was successfully dequeued, false otherwise.
 */
PR_EXPORT bool pointreceiver_dequeue_point_cloud(pointreceiver_context *ctx,
                                                 pointreceiver_pointcloud_frame *out_frame,
                                                 int timeout_ms);

/**
 * @brief Dequeues a point cloud frame from pointcaster's static frame queue
 * if one is available.
 *
 * This function attempts to dequeue a complete point cloud frame from the
 * static frame queue and fills in the provided frame structure with the
 * point count and buffer pointers. It is up for the caller to keep track of
 * their own map of source ids to latest point cloud frames.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param[out] source_id The id of the source device sending the point cloud.
 * @param[out] out_frame Pointer to a pointreceiver_pointcloud_frame structure
 * where the dequeued frame information will be stored.
 * @param timeout_ms Timeout in milliseconds to wait for a point cloud frame.
 * @return true if a frame was successfully dequeued, false otherwise.
 */
PR_EXPORT bool pointreceiver_dequeue_static_point_cloud(
    pointreceiver_context *ctx, char *out_source_id,
    pointreceiver_pointcloud_frame *out_frame, int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif // POINTRECEIVER_H
