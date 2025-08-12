from rag.context_retriever import ContextRetriever
from pprint import pprint
if __name__ == "__main__":
    context_retriever = ContextRetriever(1,8)

    # context_data = context_retriever.get_context_upto_keyframe(1, 8, "perception", 1)
    # print(con
    # qa_pair = context_retriever.get_qa_pair("perception", 56)
    # fig = context_retriever.get_annotated_images()
    # with open("test_output.png", "wb") as f:
    #     f.write(fig)

    # print("Image saved to test_output.png")
    # vehicle_data = context_retriever.get_vehicle_data_upto_sample_token()
    # pprint(vehicle_data)
    sensor_data = context_retriever.get_sensor_data_upto_sample_token()
    pprint(sensor_data)


