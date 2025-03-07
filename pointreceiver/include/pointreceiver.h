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
#elif _WIN32
#define PR_EXPORT __declspec(dllexport)
#else
#define PR_EXPORT
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
  POINTRECEIVER_MSG_TYPE_UNKNOWN                    /**< Unknown message type */
} pointreceiver_message_type;

/**
 * @brief Enumerates the possible types for parameter values.
 */
typedef enum {
  POINTRECEIVER_PARAM_VALUE_FLOAT = 0,   /**< Float value */
  POINTRECEIVER_PARAM_VALUE_INT,         /**< Integer value */
  POINTRECEIVER_PARAM_VALUE_FLOAT3,      /**< 3D float vector */
  POINTRECEIVER_PARAM_VALUE_ARRAY3F,       /**< Array of 3 floats */
  POINTRECEIVER_PARAM_VALUE_UNKNOWN        /**< Unknown parameter type */
} pointreceiver_param_value_type;

/**
 * @brief Structure representing a synchronized message.
 *
 * This structure encapsulates the message type, an identifier, and a union
 * holding the associated value.
 */
typedef struct {
  pointreceiver_message_type message_type;         /**< Type of the message */
  char id[256];                                    /**< Identifier string */
  pointreceiver_param_value_type value_type;       /**< Type of the value contained in the union */
  union {
    float float_val;                               /**< Float value */
    int int_val;                                   /**< Integer value */
    struct {
      float x, y, z;                               /**< 3D float vector */
    } float3_val;
  } value;                                         /**< Union holding the message value */
} pointreceiver_sync_message;

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
 * Attempts to dequeue a message within a specified timeout period.
 *
 * @param ctx Pointer to the PointReceiver context.
 * @param[out] out_message Pointer to a structure where the dequeued message will be stored.
 * @param timeout_ms Timeout in milliseconds to wait for a message.
 * @return true if a message was successfully dequeued, false otherwise.
 */
PR_EXPORT bool pointreceiver_dequeue_message(pointreceiver_context *ctx,
                              pointreceiver_sync_message *out_message,
                              int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif // POINTRECEIVER_H
