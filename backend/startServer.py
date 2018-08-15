import http.server
import json
import socketserver
from http.server import SimpleHTTPRequestHandler
from os import curdir
from os.path import join as pjoin

import cv2
import numpy
import pkg_resources

import base64

from backend.pyimagesearch.shapedetector import ShapeDetector
from detect_shapes import detect_shapes

INDEXFILE = pkg_resources.resource_filename(__name__, '../index.html')
BPMN_VIEWER = pkg_resources.resource_filename(__name__, '../frontend/bpmn-viewer.js')

PORT = 1337

Handler = http.server.SimpleHTTPRequestHandler


class StoreHandler(SimpleHTTPRequestHandler):

  def _set_headers(self):
    self.send_response(200)
    path = self.translate_path(self.path)
    ctype = self.guess_type(path)
    self.send_header("Content-type", 'application/json')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header("Access-Control-Allow-Headers", "x-requested-with")
    self.send_header('Access-Control-Allow-Methods', 'GET, PUT, DELETE, OPTIONS, POST')
    self.end_headers()

  sd = ShapeDetector()

  def do_GET(self):
    if self.path == '/frontend/bpmn-viewer.js':
      self.send_response(200)
      self.send_header('Content-Type', 'text/html')
      self.end_headers()
      with open(BPMN_VIEWER, 'rb') as fin:
        self.copyfile(fin, self.wfile)
    else:
      # send index.html, but don't redirect
      self.send_response(200)
      self.send_header('Content-Type', 'text/html')
      self.end_headers()
      with open(INDEXFILE, 'rb') as fin:
        self.copyfile(fin, self.wfile)

  def do_POST(self):
    print('foo')
    if self.path == '/api/image-to-bpmn':
      length = self.headers['content-length']
      data = self.rfile.read(int(length))
      # data = data.decode('base64')
      data = base64.b64decode(data)

      image_filename = pkg_resources.resource_filename('resources', 'original.png')
      with open(image_filename, 'wb') as fh:
        fh.write(data)

      self._set_headers()

      img_buffer = numpy.asarray(bytearray(data), dtype=numpy.uint8)
      original_image = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)


      # original_image = cv2.imread(retval, )
      shapes = self.sd.detect_all_shapes(original_image)
      # dict = {'foo': 'bar', 'test': {'innerfield': 'foo'}}
      self.wfile.write(json.dumps(shapes).encode('utf-8'))
      # sTest = {}
      # sTest['dummyitem'] = "Just an example of JSON"
      # self.wfile.write(json.dumps(sTest))

  def do_OPTIONS(self):
    print('option')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header("Access-Control-Allow-Headers", "x-requested-with")
    self.send_header('Access-Control-Allow-Methods', 'GET, PUT, DELETE, OPTIONS, POST')
    self.send_response(204)

  def get_mock(self):
    return {
        'startEvent1234': {
          'type': 'startEvent',
          'x': 50,
          'y': 50,
          'width': 36,
          'height': 36
        },
      'endEvent1234': {
        'type': 'endEvent',
        'x': 200,
        'y': 50,
        'width': 36,
        'height': 36
      },
      'seuqenceFlow1234': {
        'type': 'sequenceFlow',
        'waypoints': [
          { 'x': 86, 'y': 68},
          { 'x': 150, 'y': 68}
        ],
        'connects': {'from': 'startEvent1234', 'to': 'task1234'}
      },
      'task1234': {
        'type': 'task',
        'x': 150,
        'y': 46,
        'width': 100,
        'height': 80
      }
    }


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), StoreHandler) as httpd:
  httpd.allow_reuse_address = True
  print("serving at port", PORT)
  httpd.serve_forever()
