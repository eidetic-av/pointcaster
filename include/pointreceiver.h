/**
 * @file pointreceiver.h
 * @brief Public API for the PointReceiver library.
 *
 * This header provides a C API for the PointReceiver library,
 * which handles parameter updates and point cloud streams from a
 * local or remote Pointcaster instance.
 *
 * @par Ownership
 * Nothing returned by this API is owned by the caller. Point cloud buffers and
 * message payloads point into storage owned by the context, and stay valid
 * until the next dequeue of the same kind on that context, or until the
 * receiver is stopped or the context destroyed. Copy anything that needs to be
 * kept.
 *
 * @par Threading
 * A context may be created, started, stopped and destroyed from any thread, and
 * subscriptions may be changed from any thread. The two dequeue functions are
 * each expected to be driven by a single consumer thread... calling
 * pointreceiver_dequeue_point_cloud concurrently on one context can invalidate
 * buffers another caller is still reading.
 */

#pragma once
#ifndef POINTRECEIVER_H
#define POINTRECEIVER_H

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <pointreceiver_export.h>

#if defined(__cplusplus)
#define POINTRECEIVER_ALIGN(bytes) alignas(bytes)
#elif defined(_MSC_VER)
#define POINTRECEIVER_ALIGN(bytes) __declspec(align(bytes))
#else
#define POINTRECEIVER_ALIGN(bytes) __attribute__((aligned(bytes)))
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
 * @brief Result of a PointReceiver call.
 *
 * Every fallible entry point returns one of these
 */
typedef enum {
  POINTRECEIVER_OK = 0,                 /**< Call succeeded */
  POINTRECEIVER_ERROR_INVALID_ARGUMENT, /**< A required pointer was NULL, or a
                                           buffer was too small */
  POINTRECEIVER_ERROR_ALREADY_RUNNING,  /**< That receiver is already started */
  POINTRECEIVER_ERROR_NOT_RUNNING,      /**< That receiver was not started */
  POINTRECEIVER_ERROR_CONNECTION_FAILED, /**< The endpoint was rejected by the
                                            transport */
  POINTRECEIVER_ERROR_TIMEOUT,       /**< No data arrived within the timeout */
  POINTRECEIVER_ERROR_DECODE_FAILED, /**< A frame or message arrived but could
                                        not be decoded */
  POINTRECEIVER_ERROR_OUT_OF_RANGE,  /**< Index past the end of the collection
                                      */
  POINTRECEIVER_ERROR_OUT_OF_MEMORY, /**< An allocation failed */
  POINTRECEIVER_ERROR_INTERNAL /**< An unexpected internal failure; see the log
                                */
} pointreceiver_status;

/**
 * @brief Enumerates the types of messages that can be received.
 */
typedef enum {
  POINTRECEIVER_MSG_TYPE_CONNECTED = 0,             /**< Connected message */
  POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT,          /**< Client heartbeat */
  POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT_RESPONSE, /**< Response to client
                                                       heartbeat */
  POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE,  /**< Parameter update message */
  POINTRECEIVER_MSG_TYPE_PARAMETER_REQUEST, /**< Parameter request message */
  POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE,   /**< Pointcaster server endpoint
                                               update message */
  POINTRECEIVER_MSG_TYPE_UNKNOWN            /**< Unknown message type */
} pointreceiver_message_type;

/**
 * @brief Enumerates the possible types for parameter values.
 *
 * Selects the active member of pointreceiver_sync_message::value. Always set,
 * on every message: POINTRECEIVER_PARAM_VALUE_UNKNOWN means no member is live.
 */
typedef enum {
  POINTRECEIVER_PARAM_VALUE_FLOAT = 0,  /**< Float value */
  POINTRECEIVER_PARAM_VALUE_INT,        /**< Integer value */
  POINTRECEIVER_PARAM_VALUE_FLOAT2,     /**< 2D float vector */
  POINTRECEIVER_PARAM_VALUE_FLOAT3,     /**< 3D float vector */
  POINTRECEIVER_PARAM_VALUE_FLOAT4,     /**< 4D float vector */
  POINTRECEIVER_PARAM_VALUE_FLOAT2LIST, /**< List of 2D float vectors */
  POINTRECEIVER_PARAM_VALUE_FLOAT3LIST, /**< List of 3D float vectors */
  POINTRECEIVER_PARAM_VALUE_FLOAT4LIST, /**< List of 4D float vectors */
  POINTRECEIVER_PARAM_VALUE_AABBLIST,   /**< List of Axis-aligned bounding boxes
                                         */
  POINTRECEIVER_PARAM_VALUE_CONTOURSLIST,    /**< List of contours. 2D polygons
                                                stored as a list of vertex
                                                locations in pointcaster camera
                                                space */
  POINTRECEIVER_PARAM_VALUE_ENDPOINT_UPDATE, /**< Pointcaster server endpoint
                                                update */
  POINTRECEIVER_PARAM_VALUE_UNKNOWN          /**< No value */
} pointreceiver_param_value_type;

/**
 * @brief Structure representing a 2D float vector.
 */
typedef struct {
  float x; /**< X component */
  float y; /**< Y component */
} pointreceiver_float2_t;

/**
 * @brief Structure representing a 3D float vector.
 */
typedef struct {
  float x; /**< X component */
  float y; /**< Y component */
  float z; /**< Z component */
} pointreceiver_float3_t;

/**
 * @brief Structure representing a 4D float vector.
 */
typedef struct {
  float x; /**< X component */
  float y; /**< Y component */
  float z; /**< Z component */
  float w; /**< W component */
} pointreceiver_float4_t;

/**
 * @brief A single point position within a point cloud frame.
 *
 * Components are signed millimetres in pointcaster's camera space. The struct
 * mirrors pointcaster's internal position layout byte for byte, which the
 * library static-asserts at compile time; @c padding keeps it 4-byte aligned
 * and carries no meaning.
 */
typedef struct POINTRECEIVER_ALIGN(4) {
  int16_t x;       /**< X position in millimetres */
  int16_t y;       /**< Y position in millimetres */
  int16_t z;       /**< Z position in millimetres */
  int16_t padding; /**< Unused, present for alignment */
} pointreceiver_position_t;

/**
 * @brief A single point colour within a point cloud frame.
 *
 * Non-premultiplied 8-bit RGBA. Mirrors pointcaster's internal colour layout
 * byte for byte, which the library static-asserts at compile time.
 */
typedef struct {
  uint8_t r; /**< Red channel */
  uint8_t g; /**< Green channel */
  uint8_t b; /**< Blue channel */
  uint8_t a; /**< Alpha channel */
} pointreceiver_color_t;

/**
 * @brief Structure representing an Axis-Aligned Bounding Box (AABB).
 */
typedef struct {
  float min[3]; /**< Minimum coordinate (x, y, z) */
  float max[3]; /**< Maximum coordinate (x, y, z) */
} pointreceiver_aabb_t;

/**
 * @brief A borrowed list of AABBs.
 */
typedef struct {
  const pointreceiver_aabb_t *data; /**< Borrowed array of AABB values */
  size_t count;                     /**< Number of AABB values in the array */
} pointreceiver_aabb_list_t;

/**
 * @brief A borrowed list of 2D float vectors.
 */
typedef struct {
  const pointreceiver_float2_t *data; /**< Borrowed array of values */
  size_t count;                       /**< Number of values in the array */
} pointreceiver_float2_list_t;

/**
 * @brief A borrowed list of 3D float vectors.
 */
typedef struct {
  const pointreceiver_float3_t *data; /**< Borrowed array of values */
  size_t count;                       /**< Number of values in the array */
} pointreceiver_float3_list_t;

/**
 * @brief A borrowed list of 4D float vectors.
 */
typedef struct {
  const pointreceiver_float4_t *data; /**< Borrowed array of values */
  size_t count;                       /**< Number of values in the array */
} pointreceiver_float4_list_t;

/**
 * @brief A borrowed contour: a 2D polygon stored as a vertex list.
 */
typedef struct {
  const pointreceiver_float2_t *data; /**< Borrowed vertex list */
  size_t count;                       /**< Number of vertices in the contour */
} pointreceiver_contour_t;

/**
 * @brief A borrowed list of contours.
 */
typedef struct {
  const pointreceiver_contour_t *data; /**< Borrowed contours list */
  size_t count;                        /**< Number of contours in the list */
} pointreceiver_contours_list_t;

/**
 * @brief Structure representing an endpoint update from a Pointcaster server.
 */
typedef struct {
  size_t port; /**< Ephemeral port of the endpoint */
  bool active; /**< Whether the endpoint has become active or been disabled in
                  this update */
} pointreceiver_endpoint_update;

/**
 * @brief A message received from a Pointcaster instance.
 *
 * @c value_type selects the live member of @c value on every message, including
 * endpoint updates. List payloads are borrowed from the context and stay valid
 * until the next pointreceiver_dequeue_message on that context.
 */
typedef struct {
  pointreceiver_message_type message_type; /**< Type of the message */
  char id[256];                            /**< Identifier string */
  pointreceiver_param_value_type
      value_type; /**< Type of the value contained in the union */

  union {
    float float_val;                   /**< Float value */
    int int_val;                       /**< Integer value */
    pointreceiver_float2_t float2_val; /**< 2D float vector value */
    pointreceiver_float3_t float3_val; /**< 3D float vector value */
    pointreceiver_float4_t float4_val; /**< 4D float vector value */
    pointreceiver_float2_list_t
        float2_list_val; /**< List of 2D float vector values */
    pointreceiver_float3_list_t
        float3_list_val; /**< List of 3D float vector values */
    pointreceiver_float4_list_t
        float4_list_val; /**< List of 4D float vector values */
    pointreceiver_aabb_list_t aabb_list_val;         /**< List of AABB values */
    pointreceiver_contours_list_t contours_list_val; /**< List of contours */
    pointreceiver_endpoint_update
        endpoint_update_val; /**< Endpoint Update value */
  } value;                   /**< Union holding the message value */

} pointreceiver_sync_message;

/**
 * @brief A point cloud frame received from a Pointcaster instance.
 *
 * Both buffers hold @c point_count elements and are borrowed from the context.
 */
typedef struct {
  size_t point_count;                        /**< Number of points */
  const pointreceiver_position_t *positions; /**< Borrowed positions buffer */
  const pointreceiver_color_t *colours;      /**< Borrowed colours buffer */
} pointreceiver_pointcloud_frame;

/**
 * @brief Returns a short human-readable description of a status code.
 *
 * The returned string is static and always valid.
 *
 * @param status Status code to describe.
 * @return NUL-terminated description.
 */
POINTRECEIVER_EXPORT const char *
pointreceiver_status_string(pointreceiver_status status);

/**
 * @brief Creates a new PointReceiver context.
 *
 * @return A pointer to a new pointreceiver_context, or NULL on failure.
 */
POINTRECEIVER_EXPORT pointreceiver_context *pointreceiver_create_context(void);

/**
 * @brief Destroys a PointReceiver context.
 *
 * Stops any running receiver and frees all resources associated with the
 * context, invalidating every buffer previously handed out by it. Passing NULL
 * is a no-op.
 *
 * @param ctx Pointer to the context to be destroyed.
 */
POINTRECEIVER_EXPORT void
pointreceiver_destroy_context(pointreceiver_context *ctx);

/**
 * @brief Sets the context's client name.
 *
 * @param ctx Pointer to the context to target.
 * @param client_name Name for this client to send to the server on connect.
 * @return POINTRECEIVER_OK, or POINTRECEIVER_ERROR_INVALID_ARGUMENT.
 */
POINTRECEIVER_EXPORT pointreceiver_status pointreceiver_set_client_name(
    pointreceiver_context *ctx, const char *client_name);

/**
 * @brief Starts the message receiver thread.
 *
 * Connects to @p pointcaster_address and begins listening for messages.
 * A successful return means the endpoint was accepted by the transport, not
 * that a server is reachable.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param pointcaster_address The network address of the pointcaster.
 * @return POINTRECEIVER_OK, POINTRECEIVER_ERROR_INVALID_ARGUMENT,
 * POINTRECEIVER_ERROR_ALREADY_RUNNING or
 * POINTRECEIVER_ERROR_CONNECTION_FAILED.
 */
POINTRECEIVER_EXPORT pointreceiver_status pointreceiver_start_message_receiver(
    pointreceiver_context *ctx, const char *pointcaster_address);

/**
 * @brief Stops the message receiver thread.
 *
 * Blocks until the thread has ended. Invalidates any payload borrowed from
 * pointreceiver_dequeue_message.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @return POINTRECEIVER_OK, POINTRECEIVER_ERROR_INVALID_ARGUMENT or
 * POINTRECEIVER_ERROR_NOT_RUNNING.
 */
POINTRECEIVER_EXPORT pointreceiver_status
pointreceiver_stop_message_receiver(pointreceiver_context *ctx);

/**
 * @brief Reports whether the message receiver thread is running.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @return true if started and not yet stopped.
 */
POINTRECEIVER_EXPORT bool
pointreceiver_message_receiver_running(pointreceiver_context *ctx);

/**
 * @brief Dequeues a message, waiting up to @p timeout_ms for one to arrive.
 *
 * Any list payload in the returned message is borrowed from the context and
 * stays valid until the next call to this function on the same context.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param[out] out_message Structure populated with the dequeued message.
 * @param timeout_ms Timeout in milliseconds; values below zero are treated as
 * zero.
 * @return POINTRECEIVER_OK, POINTRECEIVER_ERROR_INVALID_ARGUMENT,
 * POINTRECEIVER_ERROR_TIMEOUT or POINTRECEIVER_ERROR_DECODE_FAILED.
 */
POINTRECEIVER_EXPORT pointreceiver_status pointreceiver_dequeue_message(
    pointreceiver_context *ctx, pointreceiver_sync_message *out_message,
    int timeout_ms);

/**
 * @brief Starts the point cloud receiver thread.
 *
 * Connects to @p pointcaster_address and begins receiving point cloud frames on
 * a separate socket. A successful return means the endpoint was accepted by the
 * transport, not that a server is reachable. Subscriptions registered before
 * the call are applied as soon as the thread is running.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param pointcaster_address The network address for the point cloud stream.
 * @return POINTRECEIVER_OK, POINTRECEIVER_ERROR_INVALID_ARGUMENT,
 * POINTRECEIVER_ERROR_ALREADY_RUNNING or
 * POINTRECEIVER_ERROR_CONNECTION_FAILED.
 */
POINTRECEIVER_EXPORT pointreceiver_status pointreceiver_start_point_receiver(
    pointreceiver_context *ctx, const char *pointcaster_address);

/**
 * @brief Stops the point cloud receiver thread.
 *
 * Blocks until the thread has ended. Invalidates any frame buffers borrowed
 * from pointreceiver_dequeue_point_cloud.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @return POINTRECEIVER_OK, POINTRECEIVER_ERROR_INVALID_ARGUMENT or
 * POINTRECEIVER_ERROR_NOT_RUNNING.
 */
POINTRECEIVER_EXPORT pointreceiver_status
pointreceiver_stop_point_receiver(pointreceiver_context *ctx);

/**
 * @brief Reports whether the point cloud receiver thread is running.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @return true if started and not yet stopped.
 */
POINTRECEIVER_EXPORT bool
pointreceiver_point_receiver_running(pointreceiver_context *ctx);

/**
 * @brief Subscribes to a point cloud channel by address.
 *
 * Registers interest in the channel identified by @p address. The receiver
 * thread applies the subscription to its SUB socket, which propagates upstream
 * to the publisher. Matching is exact (a NUL separator is appended internally,
 * so "foo" will not also match "foobar"). NULL or "" subscribes to all
 * channels. Safe to call from any thread and before the receiver is started;
 * pending subscriptions are applied once it is running, including after a
 * stop/start cycle.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param address Channel address, or NULL/"" for all channels.
 * @return POINTRECEIVER_OK or POINTRECEIVER_ERROR_INVALID_ARGUMENT.
 */
POINTRECEIVER_EXPORT pointreceiver_status
pointreceiver_subscribe_to_point_cloud(pointreceiver_context *ctx,
                                       const char *address);

/**
 * @brief Unsubscribes from a point cloud channel by address.
 *
 * Removes a previously registered subscription. NULL or "" removes the
 * subscribe-all subscription. Unknown addresses are ignored.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param address Channel address, or NULL/"" for all channels.
 * @return POINTRECEIVER_OK or POINTRECEIVER_ERROR_INVALID_ARGUMENT.
 */
POINTRECEIVER_EXPORT pointreceiver_status
pointreceiver_unsubscribe_from_point_cloud(pointreceiver_context *ctx,
                                           const char *address);

/**
 * @brief Dequeues a pending frame, waiting up to @p timeout_ms for one.
 *
 * The returned buffers stay valid until the next frame is dequeued, and are
 * invalidated by pointreceiver_stop_point_receiver and
 * pointreceiver_destroy_context.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param[out] out_address Buffer receiving the NUL-terminated channel address.
 * @param address_capacity Size in bytes of @p out_address.
 * @param[out] out_frame Structure populated with point count and buffers.
 * @param timeout_ms Timeout in milliseconds
 * @return POINTRECEIVER_OK, POINTRECEIVER_ERROR_INVALID_ARGUMENT,
 * POINTRECEIVER_ERROR_TIMEOUT or POINTRECEIVER_ERROR_DECODE_FAILED.
 */
POINTRECEIVER_EXPORT pointreceiver_status pointreceiver_dequeue_point_cloud(
    pointreceiver_context *ctx, char *out_address, size_t address_capacity,
    pointreceiver_pointcloud_frame *out_frame, int timeout_ms);

/**
 * @brief Returns the number of point cloud stream channel addresses observed so
 * far.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @return Number of known channel addresses.
 */
POINTRECEIVER_EXPORT size_t
pointreceiver_known_stream_address_count(pointreceiver_context *ctx);

/**
 * @brief Retrieves a known channel address by index.
 *
 * Addresses are returned in stable sorted order. @p index must be less than
 * pointreceiver_known_stream_address_count().
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param index Zero-based index of the address.
 * @param[out] out Buffer receiving the NUL-terminated address.
 * @param out_capacity Size in bytes of @p out.
 * @return POINTRECEIVER_OK, POINTRECEIVER_ERROR_INVALID_ARGUMENT or
 * POINTRECEIVER_ERROR_OUT_OF_RANGE.
 */
POINTRECEIVER_EXPORT pointreceiver_status
pointreceiver_get_known_stream_address(pointreceiver_context *ctx, size_t index,
                                       char *out, size_t out_capacity);

#ifdef __cplusplus
}
#endif

#endif // POINTRECEIVER_H
