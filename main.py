import lib.helper as helper


def main():
    video_stream_widget = helper.VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass


if __name__ == "__main__":
    main()
