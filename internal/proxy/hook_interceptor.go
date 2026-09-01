package proxy

import (
	"context"
	"strconv"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/internal/util/hookutil"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func UnaryServerHookInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		return HookInterceptor(ctx, req, GetCurUserFromContextOrDefault(ctx), info.FullMethod, handler)
	}
}

func HookInterceptor(ctx context.Context, req any, userName, fullMethod string, handler grpc.UnaryHandler) (interface{}, error) {
	hoo := hookutil.GetHook()
	var (
		newCtx   context.Context
		isMock   bool
		mockResp interface{}
		realResp interface{}
		realErr  error
		err      error
	)

	if isMock, mockResp, err = hoo.Mock(ctx, req, fullMethod); isMock {
		mlog.Info(ctx, "hook mock", mlog.String("user", userName),
			mlog.String("full method", fullMethod), mlog.Err(err))
		metrics.ProxyHookFunc.WithLabelValues(metrics.HookMock, fullMethod).Inc()
		updateProxyFunctionCallMetric(fullMethod, err)
		return mockResp, hookError(err)
	}

	if newCtx, err = hoo.Before(ctx, req, fullMethod); err != nil {
		mlog.Warn(ctx, "hook before error", mlog.String("user", userName), mlog.String("full method", fullMethod),
			GetRequestFieldWithoutSensitiveInfo(req), mlog.Err(err))
		metrics.ProxyHookFunc.WithLabelValues(metrics.HookBefore, fullMethod).Inc()
		updateProxyFunctionCallMetric(fullMethod, err)
		return nil, hookError(err)
	}
	realResp, realErr = handler(newCtx, req)
	if err = hoo.After(newCtx, realResp, realErr, fullMethod); err != nil {
		mlog.Warn(ctx, "hook after error", mlog.String("user", userName), mlog.String("full method", fullMethod),
			GetRequestFieldWithoutSensitiveInfo(req), mlog.Err(err))
		metrics.ProxyHookFunc.WithLabelValues(metrics.HookAfter, fullMethod).Inc()
		updateProxyFunctionCallMetric(fullMethod, err)
		return nil, hookError(err)
	}
	return realResp, realErr
}

// hookError is how a hook's refusal reaches the client.
//
// A plugin's error keeps the treatment it has always had: wrapped as
// InvalidArgument, because a bare error carries no classification and the SDK
// would otherwise retry a permanent refusal forever - the reason the original
// comment gave for not using merr here.
//
// An error that IS a merr sentinel already carries that classification, and
// flattening it would destroy the very thing it was chosen for: a hook that
// refuses a write with a retriable ErrServiceUnavailable means the client
// should come back, and one that withholds an RPC with ErrServiceUnimplemented
// means it never should. Those are returned as they are.
func hookError(err error) error {
	if err == nil {
		return nil
	}
	if merr.IsMilvusError(err) {
		return err
	}
	// NOTE: don't use the merr, because it will cause the wrong retry behavior in the sdk
	return status.Error(codes.InvalidArgument, "detail: "+err.Error())
}

func updateProxyFunctionCallMetric(fullMethod string, err error) {
	strs := strings.Split(fullMethod, "/")
	method := strs[len(strs)-1]
	if method == "" {
		return
	}
	status, cause := failMetricLabel(err)
	metrics.ProxyFunctionCall.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10), method, metrics.TotalLabel, metrics.CauseNA, "", "").Inc()
	metrics.ProxyFunctionCall.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10), method, status, cause, "", "").Inc()
}
