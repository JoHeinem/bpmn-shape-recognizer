import http.server
import json
import socketserver
from http.server import SimpleHTTPRequestHandler
from os import curdir
from os.path import join as pjoin

import cv2
import numpy
import pkg_resources

from detect_shapes import detect_shapes

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler


class StoreHandler(SimpleHTTPRequestHandler):
  store_path = pjoin(curdir, 'foo.wav')

  def _set_headers(self):
    self.send_response(200)
    path = self.translate_path(self.path)
    ctype = self.guess_type(path)
    self.send_header("Content-type", 'application/json')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header("Access-Control-Allow-Headers", "x-requested-with")
    self.send_header('Access-Control-Allow-Methods', 'GET, PUT, DELETE, OPTIONS, POST')
    self.end_headers()

  def do_POST(self):
    print('foo')
    if self.path == '/api/image-to-bpmn':
      length = self.headers['content-length']
      data = self.rfile.read(int(length))

      image_filename = pkg_resources.resource_filename('resources', 'foo.png')
      with open(image_filename, 'wb') as fh:
        fh.write(data)

      self._set_headers()

      img_buffer = numpy.asarray(bytearray(data), dtype=numpy.uint8)
      original_image = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
      # original_image = cv2.imread(retval, )
      detect_shapes(original_image)
      dict = {'foo': 'bar', 'test': {'innerfield': 'foo'}}
      self.wfile.write(json.dumps(dict).encode('utf-8'))
      # sTest = {}
      # sTest['dummyitem'] = "Just an example of JSON"
      # self.wfile.write(json.dumps(sTest))

  def do_OPTIONS(self):
    print('option')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header("Access-Control-Allow-Headers", "x-requested-with")
    self.send_header('Access-Control-Allow-Methods', 'GET, PUT, DELETE, OPTIONS, POST')
    self.send_response(204)


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), StoreHandler) as httpd:
  httpd.allow_reuse_address = True
  print("serving at port", PORT)
  httpd.serve_forever()
