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

#pragma once

#include "pch.h"
#include "MeshObserver.h"

namespace HoloLensForCV
{
	//
	// Saves sensor images originated on device to disk and collects sensor frame
	// metadata that will be used to create the per-sensor recording manifest CSV
	// file.
	//
	public ref class SensorFrameRecorderSink sealed
		: public ISensorFrameSink
	{
	public:
		SensorFrameRecorderSink(
			_In_ SensorType sensorType,
			_In_ Platform::String^ sensorName,
			_In_ Platform::String^ ip,
			_In_  Platform::String^ port,
			_In_  Windows::Graphics::Holographic::HolographicSpace^ holospace);

		void Start(_In_ Windows::Storage::StorageFolder^ archiveSourceFolder);

		void Stop();

		virtual void Send(_In_ SensorFrame^ sensorFrame);




	internal:
		Platform::String^ GetSensorName();

		CameraIntrinsics^ GetCameraIntrinsics();

		void ReportArchiveSourceFiles(
			_Inout_ std::vector<std::wstring>& sourceFiles);

		bool SyncStuff(bool& frameArrived, SensorType type, long long& pvtimestamp, long long timestamp);

	private:
		~SensorFrameRecorderSink();

		Platform::String^ _sensorName;

		SensorType _sensorType;

		std::mutex _sinkMutex;

		Windows::Storage::StorageFolder^ _archiveSourceFolder;

		std::unique_ptr<Io::Tarball> _bitmapTarball;
		std::unique_ptr<CsvWriter> _csvWriter;

		CameraIntrinsics^ _cameraIntrinsics;

		Windows::Foundation::DateTime _prevFrameTimestamp;


		Windows::Graphics::Imaging::SoftwareBitmap^ _lastFrame;

		char* _dest_ip;
		char* _dest_port;

		//static bool _frameArrived;

		static bool _frameArrivedVLC_LL;
		static bool _frameArrivedVLC_LF;
		static bool _frameArrivedVLC_RF;
		static bool _frameArrivedVLC_RR;
		//static bool _frameArrivedLTD;

		//static int _frames;

		//bool _frameSent;

		static long long _pv_timestampVLC_LL;
		static long long _pv_timestampVLC_LF;
		static long long _pv_timestampVLC_RF;
		static long long _pv_timestampVLC_RR;
		//static long long _pv_timestampLTD;

		Windows::Perception::Spatial::SpatialLocator^ _locator;
		Windows::Perception::Spatial::SpatialStationaryFrameOfReference^ _stationaryFrame;

		Windows::Foundation::Numerics::float3 _lastPosition;
		Windows::Foundation::Numerics::quaternion _lastOrientation;
		Windows::Foundation::Numerics::float4x4 _lastFrameToOrigin;
		Windows::Foundation::Numerics::float4x4 _lastCameraViewTransform;
		Windows::Foundation::Numerics::float4x4 _lastCameraProjectionTransform;
		long long _lastTimestamp = 0;

		//Windows::Graphics::Holographic::HolographicSpace^ _holospace;

		MeshObserver _meshObserver;
		int _frameCount;

		/*static std::mutex mtx_VLC_LL;
		static std::mutex mtx_VLC_LF;
		static std::mutex mtx_VLC_RF;
		static std::mutex mtx_VLC_RR;
		static std::mutex mtx_LTD;*/

	};
}