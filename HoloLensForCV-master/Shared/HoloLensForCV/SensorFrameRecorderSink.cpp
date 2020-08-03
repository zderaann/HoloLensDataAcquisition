//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************
#define WIN32_LEAN_AND_MEAN

#include "pch.h"
#include <stdio.h> 
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <string.h> 


// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")
#define TIMESTEP 600000  // jak dlouho maji od sebe snimky byt v 100 nanosekundach (3000000 == 0.3 sec)
#define SAVEONDEVICE true
#define NUMOFTRIANGLES 1000
#define PVALONE false
#define MESHSTEP 10
//#define IP_DEST "10.37.1.152"
//#define PORT_DEST "9099"

namespace HoloLensForCV
{
	//bool SensorFrameRecorderSink::_frameArrived;
	//int SensorFrameRecorderSink::_frames;


	bool SensorFrameRecorderSink::_frameArrivedVLC_LL;
	bool SensorFrameRecorderSink::_frameArrivedVLC_LF;
	bool SensorFrameRecorderSink::_frameArrivedVLC_RF;
	bool SensorFrameRecorderSink::_frameArrivedVLC_RR;
	//bool SensorFrameRecorderSink::_frameArrivedLTD;
	long long SensorFrameRecorderSink::_pv_timestampVLC_LL;
	long long SensorFrameRecorderSink::_pv_timestampVLC_LF;
	long long SensorFrameRecorderSink::_pv_timestampVLC_RF;
	long long SensorFrameRecorderSink::_pv_timestampVLC_RR;
	//long long SensorFrameRecorderSink::_pv_timestampLTD;


	SensorFrameRecorderSink::SensorFrameRecorderSink(
		_In_ SensorType sensorType,
		_In_ Platform::String^ sensorName,
		_In_  Platform::String^ ip,
		_In_  Platform::String^ port,
		_In_  Windows::Graphics::Holographic::HolographicSpace^ holospace)
		: _sensorType(sensorType), _sensorName(sensorName)/*, _frameSent(false)*/
	{
		const wchar_t* W = ip->Data();
		int Size = wcslen(W);
		char* CString = new char[Size + 1];
		CString[Size] = 0;
		for (int y = 0; y < Size; y++)
		{
			CString[y] = (char)W[y];
		}
		_dest_ip = (char*)CString;


		const wchar_t* CH = port->Data();
		int s = wcslen(CH);
		char* RString = new char[s + 1];
		RString[s] = 0;
		for (int y = 0; y < s; y++)
		{
			RString[y] = (char)CH[y];
		}
		_dest_port = (char*)RString;

		_frameArrivedVLC_LL = false;
		_frameArrivedVLC_LF = false;
		_frameArrivedVLC_RF = false;
		_frameArrivedVLC_RR = false;
		//_frameArrivedLTD = false;


		//_frameArrived = false;
		//_frames = 0;
		_locator = Windows::Perception::Spatial::SpatialLocator::GetDefault();
		_stationaryFrame = _locator->CreateStationaryFrameOfReferenceAtCurrentLocation();


		/*if (sensorType == SensorType::PhotoVideo) {
			_meshObserver = MeshObserver(NUMOFTRIANGLES, nullptr, nullptr, holospace);
		}

		_frameCount = MESHSTEP - 1;*/

	}

	SensorFrameRecorderSink::~SensorFrameRecorderSink()
	{
		Stop();
	}

	void SensorFrameRecorderSink::Start(
		_In_ Windows::Storage::StorageFolder^ archiveSourceFolder)
	{
		std::lock_guard<std::mutex> guard(_sinkMutex);

		// Remember the root folder for the recorded sensor meta-data.
		REQUIRES(nullptr == _archiveSourceFolder);
		_archiveSourceFolder = archiveSourceFolder;
		_meshObserver.SetArchiveFolder(archiveSourceFolder);
		// Create the tarball for the bitmap files.

		{
			wchar_t fileName[MAX_PATH] = {};
			swprintf_s(
				fileName,
				L"%s\\%s.tar",
				_archiveSourceFolder->Path->Data(),
				_sensorName->Data());
			_bitmapTarball.reset(new Io::Tarball(fileName));
		}


		// Create the csv file for the frame information.

		{
			wchar_t fileName[MAX_PATH] = {};
			swprintf_s(
				fileName,
				L"%s\\%s.csv",
				_archiveSourceFolder->Path->Data(),
				_sensorName->Data());
			_csvWriter.reset(new CsvWriter(fileName));
		}

		// Write header information to csv file.

		{
			std::vector<std::wstring> columns;

			columns.push_back(L"Timestamp");
			columns.push_back(L"ImageFileName");

			columns.push_back(L"Position.X"); columns.push_back(L"Position.Y"); columns.push_back(L"Position.Z");
			columns.push_back(L"Orientation.W"); columns.push_back(L"Orientation.X"); columns.push_back(L"Orientation.Y"); columns.push_back(L"Orientation.Z");

			columns.push_back(L"FrameToOrigin.m11"); columns.push_back(L"FrameToOrigin.m12"); columns.push_back(L"FrameToOrigin.m13"); columns.push_back(L"FrameToOrigin.m14");
			columns.push_back(L"FrameToOrigin.m21"); columns.push_back(L"FrameToOrigin.m22"); columns.push_back(L"FrameToOrigin.m23"); columns.push_back(L"FrameToOrigin.m24");
			columns.push_back(L"FrameToOrigin.m31"); columns.push_back(L"FrameToOrigin.m32"); columns.push_back(L"FrameToOrigin.m33"); columns.push_back(L"FrameToOrigin.m34");
			columns.push_back(L"FrameToOrigin.m41"); columns.push_back(L"FrameToOrigin.m42"); columns.push_back(L"FrameToOrigin.m43"); columns.push_back(L"FrameToOrigin.m44");

			columns.push_back(L"CameraViewTransform.m11"); columns.push_back(L"CameraViewTransform.m12"); columns.push_back(L"CameraViewTransform.m13"); columns.push_back(L"CameraViewTransform.m14");
			columns.push_back(L"CameraViewTransform.m21"); columns.push_back(L"CameraViewTransform.m22"); columns.push_back(L"CameraViewTransform.m23"); columns.push_back(L"CameraViewTransform.m24");
			columns.push_back(L"CameraViewTransform.m31"); columns.push_back(L"CameraViewTransform.m32"); columns.push_back(L"CameraViewTransform.m33"); columns.push_back(L"CameraViewTransform.m34");
			columns.push_back(L"CameraViewTransform.m41"); columns.push_back(L"CameraViewTransform.m42"); columns.push_back(L"CameraViewTransform.m43"); columns.push_back(L"CameraViewTransform.m44");

			columns.push_back(L"CameraProjectionTransform.m11"); columns.push_back(L"CameraProjectionTransform.m12"); columns.push_back(L"CameraProjectionTransform.m13"); columns.push_back(L"CameraProjectionTransform.m14");
			columns.push_back(L"CameraProjectionTransform.m21"); columns.push_back(L"CameraProjectionTransform.m22"); columns.push_back(L"CameraProjectionTransform.m23"); columns.push_back(L"CameraProjectionTransform.m24");
			columns.push_back(L"CameraProjectionTransform.m31"); columns.push_back(L"CameraProjectionTransform.m32"); columns.push_back(L"CameraProjectionTransform.m33"); columns.push_back(L"CameraProjectionTransform.m34");
			columns.push_back(L"CameraProjectionTransform.m41"); columns.push_back(L"CameraProjectionTransform.m42"); columns.push_back(L"CameraProjectionTransform.m43"); columns.push_back(L"CameraProjectionTransform.m44");

			_csvWriter->WriteHeader(columns);
		}
	}

	void SensorFrameRecorderSink::Stop()
	{
		std::lock_guard<std::mutex> guard(_sinkMutex);
		_bitmapTarball.reset();
		_csvWriter.reset();
		_archiveSourceFolder = nullptr;
	}

	Platform::String^ SensorFrameRecorderSink::GetSensorName()
	{
		return _sensorName;
	}

	CameraIntrinsics^ SensorFrameRecorderSink::GetCameraIntrinsics()
	{
		return _cameraIntrinsics;
	}

	void SensorFrameRecorderSink::ReportArchiveSourceFiles(
		_Inout_ std::vector<std::wstring>& sourceFiles)
	{
		wchar_t csvFileName[MAX_PATH] = {};

		swprintf_s(
			csvFileName,
			L"%s.csv",
			_sensorName->Data());

		sourceFiles.push_back(csvFileName);
	}

	bool SensorFrameRecorderSink::SyncStuff(bool& frameArrived, SensorType type, long long& timestamp, long long pvtimestamp) {
		bool res = false;
		if (frameArrived) {

			dbg::trace(L"%s: Notified from PV\n", type.ToString()->Data());
			frameArrived = false;
			res = true;
			timestamp = pvtimestamp;

		}

		return res;
	}

	bool WaitForFrameArrived(bool& frame, long long& timestamp, long long sensorframe) {
		while (frame) {
			dbg::trace(L"Waiting for frameArrived\n");
		}
		frame = true;
		timestamp = sensorframe;
		return true;
	}


	//--------------------------------------------SENDING FRAMES------------------------------------------//
	void SensorFrameRecorderSink::Send(
		SensorFrame^ sensorFrame)
	{
		long long pv_timestamp;
		bool usingLastFrame = false;
		dbg::TimerGuard timerGuard(
			L"SensorFrameRecorderSink::Send: synchrounous I/O",
			20.0 /* minimum_time_elapsed_in_milliseconds */);

		std::lock_guard<std::mutex> lockGuard(_sinkMutex);

		if (nullptr == _archiveSourceFolder)
		{
			return;
		}

		// Store a reference to the camera intrinsics.
		if (nullptr == _cameraIntrinsics)
		{
			_cameraIntrinsics = sensorFrame->SensorStreamingCameraIntrinsics;
		}

		// Avoid duplicate sensor frame recordings. + if timestamps are not 0.3s apa
		//if (_prevFrameTimestamp.Equals(sensorFrame->Timestamp)) {
		if (_sensorType == SensorType::PhotoVideo) {
			if (sensorFrame->Timestamp.UniversalTime - _prevFrameTimestamp.UniversalTime < TIMESTEP) {
				//_frameArrived = false;
				return;
			}

			_prevFrameTimestamp = sensorFrame->Timestamp;

			dbg::trace(
				L"PV: Frame arrived, notifying others\n");

			long long timestamp = sensorFrame->Timestamp.UniversalTime;
			pv_timestamp = timestamp;

			if (!PVALONE) {
				concurrency::create_task([timestamp]() -> bool {return WaitForFrameArrived(_frameArrivedVLC_LL, _pv_timestampVLC_LL, timestamp); });
				concurrency::create_task([timestamp]() -> bool {return WaitForFrameArrived(_frameArrivedVLC_LF, _pv_timestampVLC_LF, timestamp); });
				concurrency::create_task([timestamp]() -> bool {return WaitForFrameArrived(_frameArrivedVLC_RF, _pv_timestampVLC_RF, timestamp); });
				concurrency::create_task([timestamp]() -> bool {return WaitForFrameArrived(_frameArrivedVLC_RR, _pv_timestampVLC_RR, timestamp); });
				//concurrency::create_task([timestamp]() -> bool {return WaitForFrameArrived(_frameArrivedLTD, _pv_timestampLTD, timestamp); });
			}

			/*_frameCount++;

			if (_frameCount == MESHSTEP) {
				_frameCount = 0;
				auto _this = this;
				concurrency::create_task([_this, sensorFrame] {
					_this->_meshObserver.SetSpatialLocator(sensorFrame->locator);
					_this->_meshObserver.GetSurfaces(_this->_stationaryFrame->CoordinateSystem, sensorFrame->Timestamp.UniversalTime.ToString()); });
			}*/

			/*_meshObserver.SetSpatialLocator(sensorFrame->locator);
			_meshObserver.GetSurfaces(sensorFrame->coordinateSystem, sensorFrame->Timestamp.UniversalTime.ToString());*/


		}

		else if(_sensorType == SensorType::VisibleLightLeftFront || _sensorType == SensorType::VisibleLightLeftLeft || _sensorType == SensorType::VisibleLightRightFront || _sensorType == SensorType::VisibleLightRightRight) {
			//dbg::trace(L"%s: Still alive\n", _sensorType.ToString()->Data());
			if (_sensorType == SensorType::VisibleLightLeftLeft) {
				bool res = SyncStuff(_frameArrivedVLC_LL, _sensorType, pv_timestamp, _pv_timestampVLC_LL);
				if (!res) {
					_prevFrameTimestamp = sensorFrame->Timestamp;
					_lastFrame = Windows::Graphics::Imaging::SoftwareBitmap::Copy(sensorFrame->SoftwareBitmap);
					_lastPosition = sensorFrame->position;
					_lastOrientation = sensorFrame->orientation;
					_lastFrameToOrigin = sensorFrame->FrameToOrigin;
					_lastCameraViewTransform = sensorFrame->CameraViewTransform;
					_lastCameraProjectionTransform = sensorFrame->CameraProjectionTransform;
					return;
				}
			}

			else if (_sensorType == SensorType::VisibleLightLeftFront) {
				bool res = SyncStuff(_frameArrivedVLC_LF, _sensorType, pv_timestamp, _pv_timestampVLC_LF);
				if (!res) {
					_prevFrameTimestamp = sensorFrame->Timestamp;
					if (sensorFrame->SoftwareBitmap != nullptr)
						_lastFrame = Windows::Graphics::Imaging::SoftwareBitmap::Copy(sensorFrame->SoftwareBitmap);
					_lastPosition = sensorFrame->position;
					_lastOrientation = sensorFrame->orientation;
					_lastFrameToOrigin = sensorFrame->FrameToOrigin;
					_lastCameraViewTransform = sensorFrame->CameraViewTransform;
					_lastCameraProjectionTransform = sensorFrame->CameraProjectionTransform;
					return;
				}
			}
			else if (_sensorType == SensorType::VisibleLightRightFront) {
				bool res = SyncStuff(_frameArrivedVLC_RF, _sensorType, pv_timestamp, _pv_timestampVLC_RF);
				if (!res) {
					_prevFrameTimestamp = sensorFrame->Timestamp;
					if (sensorFrame->SoftwareBitmap != nullptr)
						_lastFrame = Windows::Graphics::Imaging::SoftwareBitmap::Copy(sensorFrame->SoftwareBitmap);
					_lastPosition = sensorFrame->position;
					_lastOrientation = sensorFrame->orientation;
					_lastFrameToOrigin = sensorFrame->FrameToOrigin;
					_lastCameraViewTransform = sensorFrame->CameraViewTransform;
					_lastCameraProjectionTransform = sensorFrame->CameraProjectionTransform;
					return;
				}
			}
			else if (_sensorType == SensorType::VisibleLightRightRight) {
				bool res = SyncStuff(_frameArrivedVLC_RR, _sensorType, pv_timestamp, _pv_timestampVLC_RR);
				if (!res) {
					_prevFrameTimestamp = sensorFrame->Timestamp;
					if (sensorFrame->SoftwareBitmap != nullptr)
						_lastFrame = Windows::Graphics::Imaging::SoftwareBitmap::Copy(sensorFrame->SoftwareBitmap);
					_lastPosition = sensorFrame->position;
					_lastOrientation = sensorFrame->orientation;
					_lastFrameToOrigin = sensorFrame->FrameToOrigin;
					_lastCameraViewTransform = sensorFrame->CameraViewTransform;
					_lastCameraProjectionTransform = sensorFrame->CameraProjectionTransform;
					return;
				}
			}
			/*else if (_sensorType == SensorType::LongThrowToFDepth) {
				bool res = SyncStuff(_frameArrivedLTD, _sensorType, pv_timestamp, _pv_timestampLTD);
				if (!res) {
					_prevFrameTimestamp = sensorFrame->Timestamp;
					if (sensorFrame->SoftwareBitmap != nullptr)
						_lastFrame = Windows::Graphics::Imaging::SoftwareBitmap::Copy(sensorFrame->SoftwareBitmap);
					_lastPosition = sensorFrame->position;
					_lastOrientation = sensorFrame->orientation;
					_lastFrameToOrigin = sensorFrame->FrameToOrigin;
					_lastCameraViewTransform = sensorFrame->CameraViewTransform;
					_lastCameraProjectionTransform = sensorFrame->CameraProjectionTransform;
					return;
				}
			}*/
			else {
				dbg::trace(
					L"Unsupported/unimplemented sensor!\n");
				return;
			}

			if (_sensorType != SensorType::LongThrowToFDepth && abs(_prevFrameTimestamp.UniversalTime - pv_timestamp) < abs(sensorFrame->Timestamp.UniversalTime - pv_timestamp)) {
				usingLastFrame = true;
			}
			_prevFrameTimestamp = sensorFrame->Timestamp;

		}

		if (usingLastFrame) {
			dbg::trace(L"% s: Using last frame\n", _sensorType.ToString()->Data());
		}


		/*if (sensorFrame->CoreCameraIntrinsics != nullptr)
			dbg::trace(L"%s: Focal Length: %f, %f, Principal Point: %f, %f, Radial Distortion: %f, %f, %f, Tangential Distortion: %f, %f \n", _sensorType.ToString()->Data(),
																												sensorFrame->CoreCameraIntrinsics->FocalLength.x,
																												sensorFrame->CoreCameraIntrinsics->FocalLength.y,
																												sensorFrame->CoreCameraIntrinsics->PrincipalPoint.x,
																												sensorFrame->CoreCameraIntrinsics->PrincipalPoint.y,
																												sensorFrame->CoreCameraIntrinsics->RadialDistortion.x,
																												sensorFrame->CoreCameraIntrinsics->RadialDistortion.y,
																												sensorFrame->CoreCameraIntrinsics->RadialDistortion.z,
																												sensorFrame->CoreCameraIntrinsics->TangentialDistortion.x,
																												sensorFrame->CoreCameraIntrinsics->TangentialDistortion.y
																												);*/


																												//
																												// Write the sensor frame as a bitmap to the archive.
																												//

#if DBG_ENABLE_VERBOSE_LOGGING
		dbg::trace(
			L"SensorFrameRecorderSink::Send: saving sensor frame to %s",
			bitmapPath);
#endif /* DBG_ENABLE_VERBOSE_LOGGING */
		Windows::Foundation::Numerics::float3 position;
		Windows::Foundation::Numerics::quaternion orientation;
		Windows::Foundation::Numerics::float4x4 FrameToOrigin;
		Windows::Foundation::Numerics::float4x4 CameraViewTransform;
		Windows::Foundation::Numerics::float4x4 CameraProjectionTransform;


		Windows::Graphics::Imaging::SoftwareBitmap^ softwareBitmap;
		if (usingLastFrame) {
			softwareBitmap = _lastFrame;

			position = _lastPosition;
			orientation = _lastOrientation;
			FrameToOrigin = _lastFrameToOrigin;
			CameraViewTransform = _lastCameraViewTransform;
			CameraProjectionTransform = _lastCameraProjectionTransform;
		}
		else {
			softwareBitmap = sensorFrame->SoftwareBitmap;
			position = sensorFrame->position;
			orientation = sensorFrame->orientation;
			FrameToOrigin = sensorFrame->FrameToOrigin;
			CameraViewTransform = sensorFrame->CameraViewTransform;
			CameraProjectionTransform = sensorFrame->CameraProjectionTransform;


			_lastFrame = Windows::Graphics::Imaging::SoftwareBitmap::Copy(sensorFrame->SoftwareBitmap);
			softwareBitmap = sensorFrame->SoftwareBitmap;

			_lastPosition = sensorFrame->position;
			_lastOrientation = sensorFrame->orientation;
			_lastFrameToOrigin = sensorFrame->FrameToOrigin;
			_lastCameraViewTransform = sensorFrame->CameraViewTransform;
			_lastCameraProjectionTransform = sensorFrame->CameraProjectionTransform;
		}

		// Determine metadata information about frame.

		int maxBitmapValue = 0;
		int actualBitmapWidth = softwareBitmap->PixelWidth;

		switch (softwareBitmap->BitmapPixelFormat)
		{

		case Windows::Graphics::Imaging::BitmapPixelFormat::Gray16:
			maxBitmapValue = 65535;
			break;

		case Windows::Graphics::Imaging::BitmapPixelFormat::Gray8:
			maxBitmapValue = 255;
			break;

		case Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8:
			if ((_sensorType == SensorType::VisibleLightLeftFront) ||
				(_sensorType == SensorType::VisibleLightLeftLeft) ||
				(_sensorType == SensorType::VisibleLightRightFront) ||
				(_sensorType == SensorType::VisibleLightRightRight))
			{
				maxBitmapValue = 255;
				actualBitmapWidth = actualBitmapWidth * 4;
			}
			else if (_sensorType == SensorType::PhotoVideo)
			{
				maxBitmapValue = 255;
			}
			else
			{
				ASSERT(false);
			}

			break;

		default:
			// Unsupported by PGM format. Need to update save logic
#if DBG_ENABLE_INFORMATIONAL_LOGGING
			dbg::trace(
				L"SensorFrameRecorderSink::Send: unsupported bitmap pixel format for PGM");
#endif /* DBG_ENABLE_INFORMATIONAL_LOGGING */

			ASSERT(false);
			break;
		}
		if (SAVEONDEVICE) {
			if (_sensorType == SensorType::LongThrowToFDepth && _lastTimestamp == sensorFrame->Timestamp.UniversalTime) {
				return;
			}
			else if (_sensorType == SensorType::LongThrowToFDepth) {
				_lastTimestamp = sensorFrame->Timestamp.UniversalTime;
			}

			// Determine which bitmap format to use.
			std::string bitmapFormat;
			std::wstring bitmapFileExtension;

			if (_sensorType == SensorType::PhotoVideo)
			{
				bitmapFormat = "P6";
				bitmapFileExtension = L"ppm";
			}
			else
			{
				bitmapFormat = "P5";
				bitmapFileExtension = L"pgm";
			}

			// Compose the output file name.
			wchar_t bitmapPath[MAX_PATH];
			if (_sensorType != SensorType::LongThrowToFDepth) {
				swprintf_s(
					bitmapPath, L"%s\\%020llu.%s",
					_sensorName->Data(),
					pv_timestamp,
					bitmapFileExtension.c_str());
			}
			else {
				swprintf_s(
					bitmapPath, L"%s\\%020llu.%s",
					_sensorName->Data(),
					sensorFrame->Timestamp.UniversalTime,
					bitmapFileExtension.c_str());
			}

			// Compose PGM header string.
			std::stringstream header;
			header << bitmapFormat << "\n"
				<< actualBitmapWidth << " "
				<< softwareBitmap->PixelHeight << "\n"
				<< maxBitmapValue << "\n";
			const std::string headerString = header.str();

			// Get bitmap buffer object of the frame.
			Windows::Graphics::Imaging::BitmapBuffer^ bitmapBuffer =
				softwareBitmap->LockBuffer(
					Windows::Graphics::Imaging::BitmapBufferAccessMode::Read);

			// Get raw pointer to the buffer object.
			uint32_t pixelBufferDataLength = 0;
			const uint8_t* pixelBufferData =
				Io::GetTypedPointerToMemoryBuffer<uint8_t>(
					bitmapBuffer->CreateReference(),
					pixelBufferDataLength);

			// Convert the software bitmap to raw bytes.
			std::vector<uint8_t> bitmapData;
			if (_sensorType == SensorType::PhotoVideo)
			{
				const uint32_t numPixels = softwareBitmap->PixelWidth * softwareBitmap->PixelHeight;

				// Allocate data for PGM bitmap file.
				bitmapData.reserve(headerString.size() + numPixels * 4);

				// Add PGM header data.
				bitmapData.insert(
					bitmapData.end(),
					headerString.c_str(), headerString.c_str() + headerString.size());

				bitmapData.insert(bitmapData.end(),
					pixelBufferData, pixelBufferData + pixelBufferDataLength);

				/*for (uint32_t i = 0; i < numPixels; ++i)
				{
					for (uint32_t j = 0; j < 3; ++j)
					{
						bitmapData.push_back(pixelBufferData[i * 4 + 2 - j]);
					}
				}*/
			}
			else
			{
				// Allocate data for PGM bitmap file.
				bitmapData.reserve(headerString.size() + pixelBufferDataLength);

				// Add PGM header data.
				bitmapData.insert(
					bitmapData.end(),
					headerString.c_str(), headerString.c_str() + headerString.size());

				// Add raw pixel data.
				bitmapData.insert(
					bitmapData.end(),
					pixelBufferData, pixelBufferData + pixelBufferDataLength);
			}

			dbg::trace(L"% s: Saving frame\n", _sensorType.ToString()->Data());

			// Add the bitmap to the tarball.
			_bitmapTarball->AddFile(bitmapPath, bitmapData.data(), bitmapData.size());

			//
			// Record the sensor frame meta data to the csv file.
			//


			bool writeComma = false;

			if (_sensorType != SensorType::LongThrowToFDepth) {
				_csvWriter->WriteUInt64(
					pv_timestamp, &writeComma);
			}
			else {
				_csvWriter->WriteUInt64(
					sensorFrame->Timestamp.UniversalTime, &writeComma);
			}

			{
				_csvWriter->WriteText(
					bitmapPath, &writeComma);
			}

			_csvWriter->WriteFloat3XYZ(position, &writeComma);

			_csvWriter->WriteQuaternionWXYZ(orientation, &writeComma);

			_csvWriter->WriteFloat4x4(FrameToOrigin, &writeComma);

			_csvWriter->WriteFloat4x4(CameraViewTransform, &writeComma);

			_csvWriter->WriteFloat4x4(CameraProjectionTransform, &writeComma);

			_csvWriter->EndLine();
		}
		else { //Odesilani dat na server
			//TU

			//urcim sensor, udelam "header"
			std::string sensorName;

			switch (_sensorType) {
			case SensorType::PhotoVideo:
				sensorName = "01";
				break;
			case SensorType::ShortThrowToFDepth:
				sensorName = "10";
				break;
			case SensorType::ShortThrowToFReflectivity:
				sensorName = "11";
				break;
			case SensorType::LongThrowToFDepth:
				sensorName = "12";
				break;
			case SensorType::VisibleLightLeftLeft:
				sensorName = "14";
				break;
			case SensorType::VisibleLightLeftFront:
				sensorName = "15";
				break;
			case SensorType::VisibleLightRightFront:
				sensorName = "16";
				break;
			case SensorType::VisibleLightRightRight:
				sensorName = "17";
				break;
			default:
				dbg::trace(
					L"Unknown sensor type\n");
			}


			std::stringstream header;
			//header << sensorName << sensorFrame->Timestamp.UniversalTime;
			header << sensorName << pv_timestamp;
			const std::string headerString = header.str();
			int buffsize;

			// Get bitmap buffer object of the frame.
			Windows::Graphics::Imaging::BitmapBuffer^ bitmapBuffer =
				softwareBitmap->LockBuffer(
					Windows::Graphics::Imaging::BitmapBufferAccessMode::Read);

			// Get raw pointer to the buffer object.
			uint32_t pixelBufferDataLength = 0;
			const char* pixelBufferData =
				Io::GetTypedPointerToMemoryBuffer<char>(
					bitmapBuffer->CreateReference(),
					pixelBufferDataLength);

			// Convert the software bitmap to raw bytes.
			std::vector<char> bitmapData;
			if (_sensorType == SensorType::PhotoVideo)
			{
				const uint32_t numPixels = softwareBitmap->PixelWidth * softwareBitmap->PixelHeight;

				//buffsize = headerString.size() + numPixels * 3;
				buffsize = headerString.size() + pixelBufferDataLength;
				// Allocate data for PGM bitmap file.
				bitmapData.reserve(buffsize);


				// Add PGM header data.
				bitmapData.insert(
					bitmapData.end(),
					headerString.c_str(), headerString.c_str() + headerString.size());

				bitmapData.insert(
					bitmapData.end(),
					pixelBufferData, pixelBufferData + pixelBufferDataLength);
			}
			else
			{

				buffsize = headerString.size() + pixelBufferDataLength;
				// Allocate data for PGM bitmap file.
				bitmapData.reserve(headerString.size() + pixelBufferDataLength);


				// Add PGM header data.
				bitmapData.insert(
					bitmapData.end(),
					headerString.c_str(), headerString.c_str() + headerString.size());

				// Add raw pixel data.
				bitmapData.insert(
					bitmapData.end(),
					pixelBufferData, pixelBufferData + pixelBufferDataLength);

			}


			//------------------------------------------------------------
			// Posila data


			char* imgdata = bitmapData.data();

			WSADATA wsaData;
			SOCKET ConnectSocket = INVALID_SOCKET;
			struct addrinfo* result = NULL,
				hints;
			//const char* sendbuf = "this is a test";
			int iResult;

			// Initialize Winsock
			iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
			if (iResult != 0) {
				dbg::trace(L"Socket, sensorType=%s::WSAStartup failed with error: %d\n", _sensorType.ToString()->Data(), iResult);


			}
			else {

				ZeroMemory(&hints, sizeof(hints));
				hints.ai_family = AF_INET;
				hints.ai_socktype = SOCK_STREAM;
				hints.ai_protocol = IPPROTO_TCP;

				// Resolve the server address and port
				iResult = getaddrinfo(_dest_ip, _dest_port, &hints, &result);
				if (iResult != 0) {
					dbg::trace(L"Socket, sensorType=%s::getaddrinfo failed with error: %d\n", _sensorType.ToString()->Data(), iResult);
					WSACleanup();
				}
				else {

					// Create a SOCKET for connecting to server
					ConnectSocket = socket(result->ai_family, result->ai_socktype,
						result->ai_protocol);
					if (ConnectSocket == INVALID_SOCKET) {
						dbg::trace(L"Socket, sensorType=%s::socket failed with error: %ld\n", _sensorType.ToString()->Data(), WSAGetLastError());
						WSACleanup();
					}
					else {

						// Connect to server.
						iResult = connect(ConnectSocket, result->ai_addr, (int)result->ai_addrlen);
						if (iResult == SOCKET_ERROR) {
							int err = WSAGetLastError();
							closesocket(ConnectSocket);
							ConnectSocket = INVALID_SOCKET;
							dbg::trace(L"Socket, sensorType=%s::Unable to connect to server! error: %ld\n", _sensorType.ToString()->Data(), err);
							WSACleanup();
						}
						else {


							freeaddrinfo(result);


							// Send an initial buffer
							iResult = send(ConnectSocket, imgdata, buffsize, 0);  //Exception thrown at 0x5865FBA0 (ucrtbased.dll) in Recorder.exe: 0xC0000005: Access violation reading location 0x0E8DF000.
							if (iResult == SOCKET_ERROR) {
								dbg::trace(L"Socket, sensorType=%s::send failed with error: %d\n", _sensorType.ToString()->Data(), WSAGetLastError());
								closesocket(ConnectSocket);
								WSACleanup();
							}
							else {

								dbg::trace(L"Socket, sensorType=%s::Bytes Sent: %ld out of %d\n", _sensorType.ToString()->Data(), iResult, buffsize);

								// shutdown the connection since no more data will be sent
								iResult = shutdown(ConnectSocket, SD_SEND);
								if (iResult == SOCKET_ERROR) {
									dbg::trace(L"Socket, sensorType=%s::shutdown failed with error: %d\n", _sensorType.ToString()->Data(), WSAGetLastError());
									closesocket(ConnectSocket);
									WSACleanup();

								}
								else {
									// cleanup
									closesocket(ConnectSocket);
									WSACleanup();
								}
							}
						}
					}
				}
			}
		}
	}
}