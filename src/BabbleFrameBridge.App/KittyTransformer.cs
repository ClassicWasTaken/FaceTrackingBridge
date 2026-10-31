// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0
using System.Numerics;

namespace BabbleFrameBridge.App;

/// <summary>
/// Bounded, CPU-only causal Transformer inference. Matrices are exported in
/// input-major order. No tensors, recordings, or external runtime are retained.
/// </summary>
internal sealed class KittyTransformer
{
    private const int Width = KittyBrain.ModelWidth, Context = KittyBrain.ContextLength;
    private const int Heads = KittyBrain.HeadCount, HeadWidth = Width / Heads;
    private readonly float[] _state = new float[Context * Width];
    private readonly float[] _normalized = new float[Context * Width];
    private readonly float[] _qkv = new float[Context * Width * 3];
    private readonly float[] _attended = new float[Context * Width];
    private readonly float[] _projected = new float[Context * Width];
    private readonly float[] _feedForward = new float[Context * KittyBrain.FeedForwardWidth];

    internal void Reset()
    {
        Array.Clear(_state); Array.Clear(_normalized); Array.Clear(_qkv);
        Array.Clear(_attended); Array.Clear(_projected); Array.Clear(_feedForward);
    }

    internal void Evaluate(ReadOnlySpan<float> sequence, int length, Span<double> behavior,
        Span<double> attention, int outputPosition = -1)
    {
        if (length is < 1 or > Context || sequence.Length != length * KittyBrain.InputCount)
            throw new ArgumentException("The Transformer requires one to 32 complete observation tokens.", nameof(sequence));
        if (behavior.Length != KittyBrain.ActionCount || attention.Length != KittyBrain.AttentionCount)
            throw new ArgumentException("The Transformer output heads have fixed sizes.");
        if (outputPosition == -1) outputPosition = length - 1;
        if (outputPosition < 0 || outputPosition >= length) throw new ArgumentOutOfRangeException(nameof(outputPosition));

        Dense(sequence, _state.AsSpan(0, length * Width), length, KittyBrain.InputCount, Width,
            KittyModel.InputWeights, KittyModel.InputBias);
        for (var index = 0; index < length * Width; index++) _state[index] += KittyModel.PositionEncoding[index];
        Block(length, KittyModel.Block0Norm1Scale, KittyModel.Block0Norm1Bias, KittyModel.Block0QkvWeights,
            KittyModel.Block0QkvBias, KittyModel.Block0ProjectionWeights, KittyModel.Block0ProjectionBias,
            KittyModel.Block0Norm2Scale, KittyModel.Block0Norm2Bias, KittyModel.Block0FeedForwardWeights,
            KittyModel.Block0FeedForwardBias, KittyModel.Block0FeedForwardOutputWeights, KittyModel.Block0FeedForwardOutputBias);
        Block(length, KittyModel.Block1Norm1Scale, KittyModel.Block1Norm1Bias, KittyModel.Block1QkvWeights,
            KittyModel.Block1QkvBias, KittyModel.Block1ProjectionWeights, KittyModel.Block1ProjectionBias,
            KittyModel.Block1Norm2Scale, KittyModel.Block1Norm2Bias, KittyModel.Block1FeedForwardWeights,
            KittyModel.Block1FeedForwardBias, KittyModel.Block1FeedForwardOutputWeights, KittyModel.Block1FeedForwardOutputBias);
        Normalize(_state.AsSpan(outputPosition * Width, Width), _normalized.AsSpan(0, Width),
            1, KittyModel.FinalNormScale, KittyModel.FinalNormBias);
        SoftmaxHead(_normalized.AsSpan(0, Width), behavior, KittyModel.OutputWeights, KittyModel.OutputBias);
        SoftmaxHead(_normalized.AsSpan(0, Width), attention, KittyModel.AttentionWeights, KittyModel.AttentionBias);
    }

    private void Block(int length, float[] norm1Scale, float[] norm1Bias, float[] qkvWeights, float[] qkvBias,
        float[] projectionWeights, float[] projectionBias, float[] norm2Scale, float[] norm2Bias,
        float[] feedForwardWeights, float[] feedForwardBias, float[] feedForwardOutputWeights, float[] feedForwardOutputBias)
    {
        var count = length * Width;
        Normalize(_state.AsSpan(0, count), _normalized.AsSpan(0, count), length, norm1Scale, norm1Bias);
        Dense(_normalized, _qkv, length, Width, Width * 3, qkvWeights, qkvBias);
        CausalAttention(length);
        Dense(_attended, _projected, length, Width, Width, projectionWeights, projectionBias);
        for (var index = 0; index < count; index++) _state[index] += _projected[index];
        Normalize(_state.AsSpan(0, count), _normalized.AsSpan(0, count), length, norm2Scale, norm2Bias);
        Dense(_normalized, _feedForward, length, Width, KittyBrain.FeedForwardWidth, feedForwardWeights, feedForwardBias);
        for (var index = 0; index < length * KittyBrain.FeedForwardWidth; index++)
        {
            var value = _feedForward[index];
            _feedForward[index] = (float)(.5 * value * (1 + Math.Tanh(.7978845608028654 * (value + .044715 * value * value * value))));
        }
        Dense(_feedForward, _projected, length, KittyBrain.FeedForwardWidth, Width, feedForwardOutputWeights, feedForwardOutputBias);
        for (var index = 0; index < count; index++) _state[index] += _projected[index];
    }

    private void CausalAttention(int length)
    {
        // Each head sees only its own present and preceding positions. Keeping
        // the strict upper triangle out of both softmax and weighted values is
        // the causal mask; padding and future observations cannot influence it.
        Span<double> scores = stackalloc double[Context];
        _attended.AsSpan(0, length * Width).Clear();
        for (var position = 0; position < length; position++)
        for (var head = 0; head < Heads; head++)
        {
            var query = position * Width * 3 + head * HeadWidth;
            var largest = double.NegativeInfinity;
            for (var source = 0; source <= position; source++)
            {
                var key = source * Width * 3 + Width + head * HeadWidth;
                double dot = 0;
                for (var element = 0; element < HeadWidth; element++) dot += _qkv[query + element] * _qkv[key + element];
                var score = dot / Math.Sqrt(HeadWidth);
                scores[source] = score; largest = Math.Max(largest, score);
            }
            double sum = 0;
            for (var source = 0; source <= position; source++) sum += scores[source] = Math.Exp(scores[source] - largest);
            var output = position * Width + head * HeadWidth;
            for (var source = 0; source <= position; source++)
            {
                var value = source * Width * 3 + Width * 2 + head * HeadWidth;
                var weight = (float)(scores[source] / sum);
                for (var element = 0; element < HeadWidth; element++) _attended[output + element] += weight * _qkv[value + element];
            }
        }
    }

    private static void Normalize(ReadOnlySpan<float> input, Span<float> output, int rows, float[] scale, float[] bias)
    {
        for (var row = 0; row < rows; row++)
        {
            var offset = row * Width;
            double mean = 0, variance = 0;
            for (var index = 0; index < Width; index++) mean += input[offset + index];
            mean /= Width;
            for (var index = 0; index < Width; index++)
            { var difference = input[offset + index] - mean; variance += difference * difference; }
            var inverse = 1 / Math.Sqrt(variance / Width + 1e-5);
            for (var index = 0; index < Width; index++)
                output[offset + index] = (float)((input[offset + index] - mean) * inverse * scale[index] + bias[index]);
        }
    }

    private static void Dense(ReadOnlySpan<float> input, Span<float> output, int rows, int inputs, int outputs, float[] weights, float[] bias)
    {
        for (var row = 0; row < rows; row++)
        {
            var target = output.Slice(row * outputs, outputs);
            bias.AsSpan().CopyTo(target);
            for (var index = 0; index < inputs; index++)
            {
                var value = input[row * inputs + index];
                var start = index * outputs;
                var column = 0;
                // Vectorization bounds CPU work at full context without adding a
                // platform-specific native dependency or per-inference arrays.
                if (Vector.IsHardwareAccelerated)
                {
                    var factor = new Vector<float>(value);
                    for (; column <= outputs - Vector<float>.Count; column += Vector<float>.Count)
                        (new Vector<float>(target.Slice(column)) + factor * new Vector<float>(weights.AsSpan(start + column)))
                            .CopyTo(target.Slice(column));
                }
                for (; column < outputs; column++) target[column] += value * weights[start + column];
            }
        }
    }

    private static void SoftmaxHead(ReadOnlySpan<float> input, Span<double> output, float[] weights, float[] bias)
    {
        var largest = double.NegativeInfinity;
        for (var index = 0; index < output.Length; index++)
        {
            double value = bias[index];
            for (var source = 0; source < Width; source++) value += input[source] * weights[source * output.Length + index];
            output[index] = value; largest = Math.Max(largest, value);
        }
        double sum = 0;
        for (var index = 0; index < output.Length; index++) sum += output[index] = Math.Exp(output[index] - largest);
        for (var index = 0; index < output.Length; index++) output[index] /= sum;
    }
}
