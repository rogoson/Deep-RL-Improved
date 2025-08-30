import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import pickle
from main.environments.TimeSeriesEnvironment import TimeSeriesEnvironment
import yaml

# global thread reference and lock for video generation
generationThread = None
generationLock = threading.Lock()

# path to the generated video (updated by the thread)
latestVideoPath = None


class RestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global generationThread, latestVideoPath

        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        elif self.path == "/generateAnimation":
            # so users have patience when a video is being generated
            with generationLock:  # one at a time boys
                if generationThread and generationThread.is_alive():
                    # if there's already a generation thread and it's active
                    self.send_response(429)
                    self.end_headers()
                    self.wfile.write(b"A video is already being generated, be patient.")
                    return

                # start a new thread for video generation
                generationThread = threading.Thread(
                    target=self.generateVideo, daemon=False
                )
                generationThread.start()

            self.send_response(202)  # Accepted
            self.end_headers()
            self.wfile.write(b"Video generation started")

        elif self.path == "/getVideo":
            # serve the most recently generated video
            if latestVideoPath and Path(latestVideoPath).exists():
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header(
                    "Content-Length", str(Path(latestVideoPath).stat().st_size)
                )
                self.end_headers()
                with open(latestVideoPath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Video not ready yet")

        else:
            self.send_response(404)
            self.end_headers()

    def generateVideo(self):
        global latestVideoPath
        try:
            videoEnvironment = TimeSeriesEnvironment(None, None, None, None)
            agentDetailsPath = (
                Path(__file__).resolve().parent.parent
                / "animations"
                / "agentAnimationDetails.yaml"
            )

            if agentDetailsPath.exists():
                videoConfig = yaml.safe_load(agentDetailsPath)

                with Path(videoConfig["path"]).open("rb") as f:
                    loadedState = pickle.load(f)

                videoEnvironment.__dict__.update(loadedState)

                videoPath = videoEnvironment.generateAnimation(
                    agentType=videoConfig["agentType"],
                    stage=videoConfig["stage"],
                    index=videoConfig["index"],
                    featureExtractor=videoConfig["featureExtractor"],
                )

                video = Path(videoPath) / "portfolio_animation.mp4"
                if video.exists():
                    latestVideoPath = str(video)  # update global reference
                else:
                    print("Video not found after generation.")
            else:
                print("No best agent yet - agentAnimationDetails.yaml missing.")

        except Exception as e:
            print(f"Some error occurred during video generation: {str(e)}")


def startServer():
    server = HTTPServer(("0.0.0.0", 8080), RestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
