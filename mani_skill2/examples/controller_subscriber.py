import zmq
import multiprocessing as mp
from mani_skill2.examples.vr_controller_state import parse_controller_state
import time

def vr_subscriber(subscriber_address, topic, queue):
    # Define the subscriber's address
    # subscriber_address = VR_TCP_ADDRESS

    # create socket and connect to the publisher
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(subscriber_address)

    # subscribe to desired topic
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    print("Start Listening...")

    end_check = False
    while not end_check:

        [received_topic, received_data] = socket.recv_multipart()
        parsed_data = received_data.decode("utf-8")
        parsed_data = parse_controller_state(parsed_data)

        if parsed_data.left_y and parsed_data.right_b:
            end_check = True
            print("End of the subscriber")
        # time.sleep(0.1)
        queue.put(parsed_data)



# # Define the subscriber's address
# subscriber_address = VR_TCP_ADDRESS

# # create socket and connect to the publisher
# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# socket.connect(subscriber_address)

# # subscribe to desired topic
# socket.setsockopt_string(zmq.SUBSCRIBE, VR_TOPIC)

# print("Start Listening...")

# end_check = False
# while not end_check:

#     [received_topic, received_data] = socket.recv_multipart()
#     parsed_data = received_data.decode("utf-8")
#     parsed_data = parse_controller_state(parsed_data)
#     print(f"Data: {parsed_data}")

#     if parsed_data.left_x and parsed_data.right_a:
#         end_check = True
#         print("End of the subscriber")