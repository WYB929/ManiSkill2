import zmq
import multiprocessing as mp
from mani_skill2.examples.vr_controller_state import parse_controller_state
from mani_skill2.examples import VR_TCP_ADDRESS, VR_TOPIC
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
    socket.setsockopt(zmq.CONFLATE, 1)

    print("Start Listening...")

    end_check = False
    while not end_check:

        [received_topic, received_data] = socket.recv_multipart()
        parsed_data = received_data.decode("utf-8")
        parsed_data = parse_controller_state(parsed_data)

        if parsed_data.left_y and parsed_data.right_b:
            end_check = True
            print("End of the subscriber")
        if parsed_data.right_a:
            print("Right Telelop Seleted")
        elif parsed_data.left_x:
            print("Left Telelop Seleted")
        queue.put(parsed_data)

# test code
if __name__ == "__main__":
    # Define the subscriber's address
    subscriber_address = VR_TCP_ADDRESS

    # create socket and connect to the publisher
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(subscriber_address)

    # subscribe to desired topic
    socket.setsockopt_string(zmq.SUBSCRIBE, VR_TOPIC)
    socket.setsockopt(zmq.CONFLATE, 1)

    print("Start Listening...")

    end_check = False
    while not end_check:
        
        import timeit; start = timeit.default_timer()
        [received_topic, received_data] = socket.recv_multipart()
        parsed_data = received_data.decode("utf-8")
        parsed_data = parse_controller_state(parsed_data)
        print(f"Data: {parsed_data}")
        if parsed_data.left_y and parsed_data.right_b:
            end_check = True
            print("End of the subscriber")
        print(f"Time: {timeit.default_timer() - start}")