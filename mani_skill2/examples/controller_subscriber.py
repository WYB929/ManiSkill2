import zmq
import multiprocessing as mp
from mani_skill2.examples.vr_controller_state import parse_controller_state
from mani_skill2.examples import VR_TCP_ADDRESS, VR_TOPIC
import time

# helper function to update the queue, only keep the latest data
def update_queue(queue, data, lock):
    with lock:
        if queue.empty():
            queue.put(data)
        else: 
            queue.get()
            queue.put(data)

def get_data(lock, queue):
    with lock:
        if not queue.empty():
            return queue.get()


def vr_subscriber(subscriber_address, topic, queue, lock):
    # Define the subscriber's address
    # create socket and connect to the publisher
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # subscribe to desired topic
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(subscriber_address)
    

    print("Start Listening...")
    index = 0
    end_check = False
    while not end_check:

        [received_topic, received_data] = socket.recv_multipart()
        parsed_data = received_data.decode("utf-8")
        parsed_data = parse_controller_state(parsed_data)
        index += 1
        print("index: ", index)
        

        if parsed_data.left_x and parsed_data.right_a:
            end_check = True
            print("End of the subscriber")
        update_queue(queue, parsed_data, lock)

# test code
if __name__ == "__main__":
    # Define the subscriber's address
    subscriber_address = VR_TCP_ADDRESS

    # create socket and connect to the publisher
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # subscribe to desired topic
    socket.setsockopt_string(zmq.SUBSCRIBE, VR_TOPIC)
    socket.connect(subscriber_address)
    

    print("Test Start Listening...")

    end_check = False
    while not end_check:
        
        [received_topic, received_data] = socket.recv_multipart()
        parsed_data = received_data.decode("utf-8")
        parsed_data = parse_controller_state(parsed_data)
        print(f"Data: {parsed_data}")
        if parsed_data.left_y and parsed_data.right_b:
            end_check = True
            print("End of the subscriber")