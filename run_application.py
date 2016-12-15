import os
import tornado.ioloop
import tornado.httpserver
import tornado.escape
from tornado.options import define, options
from application.server import Application

# Define command line arguments
define("port", default=3000, help="run on the given port", type=int)


def main():
    # tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    port = int(os.environ.get("PORT", options.port))
    print("server is running on port {0}".format(port))
    http_server.listen(port)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(ex)
