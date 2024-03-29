# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: segmenter_service.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17segmenter_service.proto\x1a\x1bgoogle/protobuf/empty.proto\"D\n\x06Mask2D\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05\x64type\x18\x02 \x01(\t\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\"5\n\x07NdArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05\x64type\x18\x02 \x01(\t\x12\r\n\x05shape\x18\x03 \x03(\x05\"!\n\x10SensitivityValue\x12\r\n\x05value\x18\x01 \x01(\x02\"T\n\nMarkerMask\x12\x15\n\x04mask\x18\x01 \x01(\x0b\x32\x07.Mask2D\x12\x0e\n\x06marker\x18\x02 \x01(\t\x12\x1f\n\x04sens\x18\x03 \x01(\x0b\x32\x11.SensitivityValue\"}\n\x0cMethodParams\x12\x13\n\x0bmethod_name\x18\x01 \x01(\t\x12)\n\x06params\x18\x02 \x03(\x0b\x32\x19.MethodParams.ParamsEntry\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\"S\n\x10NewMethodRequest\x12$\n\rmethod_params\x18\x01 \x01(\x0b\x32\r.MethodParams\x12\x19\n\x11save_current_mask\x18\x02 \x01(\x08\"\xc6\x01\n\x10LoadImageRequest\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x1d\n\x0bimage_array\x18\x02 \x01(\x0b\x32\x08.NdArray\x12/\n\x07markers\x18\x03 \x03(\x0b\x32\x1e.LoadImageRequest.MarkersEntry\x12$\n\rmethod_params\x18\x04 \x01(\x0b\x32\r.MethodParams\x1a.\n\x0cMarkersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x32\x8a\x03\n\tSegmenter\x12\x38\n\tloadImage\x12\x11.LoadImageRequest\x1a\x16.google.protobuf.Empty\"\x00\x12\x38\n\tnewMethod\x12\x11.NewMethodRequest\x1a\x16.google.protobuf.Empty\"\x00\x12:\n\x0bsendNewSens\x12\x11.SensitivityValue\x1a\x16.google.protobuf.Empty\"\x00\x12\x37\n\x0esendMarkerMask\x12\x0b.MarkerMask\x1a\x16.google.protobuf.Empty\"\x00\x12\x34\n\x0fgetMarkedMask2D\x12\x16.google.protobuf.Empty\x1a\x07.Mask2D\"\x00\x12/\n\ngetRegions\x12\x16.google.protobuf.Empty\x1a\x07.Mask2D\"\x00\x12-\n\x08popState\x12\x16.google.protobuf.Empty\x1a\x07.Mask2D\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'segmenter_service_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _METHODPARAMS_PARAMSENTRY._options = None
  _METHODPARAMS_PARAMSENTRY._serialized_options = b'8\001'
  _LOADIMAGEREQUEST_MARKERSENTRY._options = None
  _LOADIMAGEREQUEST_MARKERSENTRY._serialized_options = b'8\001'
  _MASK2D._serialized_start=56
  _MASK2D._serialized_end=124
  _NDARRAY._serialized_start=126
  _NDARRAY._serialized_end=179
  _SENSITIVITYVALUE._serialized_start=181
  _SENSITIVITYVALUE._serialized_end=214
  _MARKERMASK._serialized_start=216
  _MARKERMASK._serialized_end=300
  _METHODPARAMS._serialized_start=302
  _METHODPARAMS._serialized_end=427
  _METHODPARAMS_PARAMSENTRY._serialized_start=382
  _METHODPARAMS_PARAMSENTRY._serialized_end=427
  _NEWMETHODREQUEST._serialized_start=429
  _NEWMETHODREQUEST._serialized_end=512
  _LOADIMAGEREQUEST._serialized_start=515
  _LOADIMAGEREQUEST._serialized_end=713
  _LOADIMAGEREQUEST_MARKERSENTRY._serialized_start=667
  _LOADIMAGEREQUEST_MARKERSENTRY._serialized_end=713
  _SEGMENTER._serialized_start=716
  _SEGMENTER._serialized_end=1110
# @@protoc_insertion_point(module_scope)
