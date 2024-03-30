TABLE = [
  {
    "domain": "",
    "name": "Abs",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Abs",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Acos",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Acosh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Add",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      7,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Add",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Add",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Affine",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "AffineGrid",
    "input_types": [
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "And",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)"
      ],
      "T1": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ArgMax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ArgMax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ArgMax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ArgMin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ArgMin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ArgMin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Asin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Asinh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Atan",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Atanh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "AveragePool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      7,
      9
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "AveragePool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      10,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "AveragePool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "AveragePool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BatchNormalization",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "T",
      "T",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      7,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BatchNormalization",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "T",
      "T",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      9,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BatchNormalization",
    "input_types": [
      "T",
      "T",
      "T",
      "U",
      "U"
    ],
    "outputs_types": [
      "T",
      "U",
      "U"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "U": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      14,
      14
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BatchNormalization",
    "input_types": [
      "T",
      "T1",
      "T1",
      "T2",
      "T2"
    ],
    "outputs_types": [
      "T",
      "T2",
      "T2"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      15,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BitShift",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BitwiseAnd",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BitwiseNot",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BitwiseOr",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BitwiseXor",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "BlackmanWindow",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      17,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Cast",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Cast",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ]
    },
    "version_range": [
      13,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Cast",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Ceil",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Ceil",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Celu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      12,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Clip",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Clip",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Clip",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int64)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Clip",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int64)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Col2Im",
    "input_types": [
      "T",
      "tensor(int64)",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Compress",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "T1": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      9,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Compress",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "T1": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Concat",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      4,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Concat",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Concat",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ConcatFromSequence",
    "input_types": [
      "S"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ConstantOfShape",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int64)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      9,
      19
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ConstantOfShape",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int64)"
      ],
      "T2": [
        "tensor(bfloat16)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Conv",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Conv",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ConvInteger",
    "input_types": [
      "T1",
      "T2",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)"
      ],
      "T2": [
        "tensor(uint8)"
      ],
      "T3": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      10,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ConvTranspose",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ConvTranspose",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Cos",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Cosh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Crop",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "CumSum",
    "input_types": [
      "T",
      "T2"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "CumSum",
    "input_types": [
      "T",
      "T2"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DFT",
    "input_types": [
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      17,
      19
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DFT",
    "input_types": [
      "T1",
      "T2",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DepthToSpace",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DepthToSpace",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DepthToSpace",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DequantizeLinear",
    "input_types": [
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      10,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DequantizeLinear",
    "input_types": [
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      13,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DequantizeLinear",
    "input_types": [
      "T1",
      "T2",
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int32)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Det",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Div",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      7,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Div",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Div",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Dropout",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      7,
      9
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Dropout",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T",
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      10,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Dropout",
    "input_types": [
      "T",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T",
      "T2"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(float)",
        "tensor(double)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Dropout",
    "input_types": [
      "T",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T",
      "T2"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(float)",
        "tensor(double)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DynamicQuantizeLinear",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2",
      "tensor(float)",
      "T2"
    ],
    "type_constraints": {
      "T2": [
        "tensor(uint8)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "DynamicSlice",
    "input_types": [
      "T",
      "Tind",
      "Tind",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Einsum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      12,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Elu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Equal",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      7,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Equal",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Equal",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      13,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Equal",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(string)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Erf",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Erf",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Exp",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Exp",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Expand",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(bool)",
        "tensor(float16)",
        "tensor(string)"
      ]
    },
    "version_range": [
      8,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Expand",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(bool)",
        "tensor(float16)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "EyeLike",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(int64)",
        "tensor(int32)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(int64)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Flatten",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Flatten",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      9,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Flatten",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Flatten",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Floor",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Floor",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GRU",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T"
    ],
    "outputs_types": [
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      7,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GRU",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T"
    ],
    "outputs_types": [
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Gather",
    "input_types": [
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Gather",
    "input_types": [
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Gather",
    "input_types": [
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GatherElements",
    "input_types": [
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GatherElements",
    "input_types": [
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GatherND",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "indices": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GatherND",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "indices": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GatherND",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "indices": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Gemm",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      7,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Gemm",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      9,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Gemm",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Gemm",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GlobalAveragePool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GlobalLpPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      2,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GlobalMaxPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Greater",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      7,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Greater",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Greater",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GreaterOrEqual",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      12,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GreaterOrEqual",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      16,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GridSample",
    "input_types": [
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(float)"
      ]
    },
    "version_range": [
      16,
      19
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "GridSample",
    "input_types": [
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "HammingWindow",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      17,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "HannWindow",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      17,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "HardSigmoid",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Hardmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Hardmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Hardmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Identity",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Identity",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Identity",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      14,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Identity",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ]
    },
    "version_range": [
      16,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Identity",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "seq(tensor(float8e4m3fn))",
        "seq(tensor(float8e4m3fnuz))",
        "seq(tensor(float8e5m2))",
        "seq(tensor(float8e5m2fnuz))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "If",
    "input_types": [
      "B"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "If",
    "input_types": [
      "B"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "If",
    "input_types": [
      "B"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      13,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "If",
    "input_types": [
      "B"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ]
    },
    "version_range": [
      16,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "If",
    "input_types": [
      "B"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "seq(tensor(float8e4m3fn))",
        "seq(tensor(float8e4m3fnuz))",
        "seq(tensor(float8e5m2))",
        "seq(tensor(float8e5m2fnuz))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ImageScaler",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "InstanceNormalization",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "IsInf",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      10,
      19
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "IsInf",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "T2": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "IsNaN",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "IsNaN",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      13,
      19
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "IsNaN",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "T2": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LRN",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LRN",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LSTM",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      7,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LSTM",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LayerNormalization",
    "input_types": [
      "T",
      "V",
      "V"
    ],
    "outputs_types": [
      "V",
      "U",
      "U"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "U": [
        "tensor(float)",
        "tensor(double)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      16
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LayerNormalization",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "U",
      "U"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "U": [
        "tensor(float)",
        "tensor(float)"
      ]
    },
    "version_range": [
      17,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LeakyRelu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LeakyRelu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      16,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Less",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      7,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Less",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Less",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LessOrEqual",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      12,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LessOrEqual",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      16,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Log",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Log",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LogSoftmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LogSoftmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LogSoftmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Loop",
    "input_types": [
      "I",
      "B",
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "I": [
        "tensor(int64)"
      ],
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Loop",
    "input_types": [
      "I",
      "B",
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "I": [
        "tensor(int64)"
      ],
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Loop",
    "input_types": [
      "I",
      "B",
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "I": [
        "tensor(int64)"
      ],
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      13,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Loop",
    "input_types": [
      "I",
      "B",
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "I": [
        "tensor(int64)"
      ],
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ]
    },
    "version_range": [
      16,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Loop",
    "input_types": [
      "I",
      "B",
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "I": [
        "tensor(int64)"
      ],
      "B": [
        "tensor(bool)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "seq(tensor(float8e4m3fn))",
        "seq(tensor(float8e4m3fnuz))",
        "seq(tensor(float8e5m2))",
        "seq(tensor(float8e5m2fnuz))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LpNormalization",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LpPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      2,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LpPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "LpPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MatMul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MatMul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int64)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MatMul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int64)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MatMulInteger",
    "input_types": [
      "T1",
      "T2",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int8)"
      ],
      "T3": [
        "tensor(int32)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      10,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Max",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      7
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Max",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      8,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Max",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Max",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MaxPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      7
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MaxPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T",
      "I"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "I": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      8,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MaxPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T",
      "I"
    ],
    "type_constraints": {
      "T": [
        "tensor(double)",
        "tensor(float)",
        "tensor(int8)",
        "tensor(uint8)"
      ],
      "I": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      12,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MaxRoiPool",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MaxUnpool",
    "input_types": [
      "T1",
      "T2",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      9,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MaxUnpool",
    "input_types": [
      "T1",
      "T2",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mean",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      7
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mean",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      8,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mean",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MeanVarianceNormalization",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MeanVarianceNormalization",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MeanVarianceNormalization",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "MelWeightMatrix",
    "input_types": [
      "T1",
      "T1",
      "T1",
      "T2",
      "T2"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(float)"
      ],
      "T3": [
        "tensor(float)",
        "tensor(double)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      17,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Min",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      7
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Min",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      8,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Min",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Min",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mod",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)"
      ]
    },
    "version_range": [
      10,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mod",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      7,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Mul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Multinomial",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Neg",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Neg",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "NonMaxSuppression",
    "input_types": [
      "tensor(float)",
      "tensor(float)",
      "tensor(int64)",
      "tensor(float)",
      "tensor(float)"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {},
    "version_range": [
      10,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "NonMaxSuppression",
    "input_types": [
      "tensor(float)",
      "tensor(float)",
      "tensor(int64)",
      "tensor(float)",
      "tensor(float)"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {},
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "NonZero",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "NonZero",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Not",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "OneHot",
    "input_types": [
      "T1",
      "T2",
      "T3"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int64)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int32)"
      ],
      "T3": [
        "tensor(int64)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(string)",
        "tensor(float)",
        "tensor(int32)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "OneHot",
    "input_types": [
      "T1",
      "T2",
      "T3"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int64)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(int32)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int32)"
      ],
      "T3": [
        "tensor(int64)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(string)",
        "tensor(float)",
        "tensor(int32)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Optional",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "O"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ],
      "O": [
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ]
    },
    "version_range": [
      15,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "OptionalGetElement",
    "input_types": [
      "O"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "O": [
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      15,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "OptionalGetElement",
    "input_types": [
      "O"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "O": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "OptionalHasElement",
    "input_types": [
      "O"
    ],
    "outputs_types": [
      "B"
    ],
    "type_constraints": {
      "O": [
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ],
      "B": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      15,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "OptionalHasElement",
    "input_types": [
      "O"
    ],
    "outputs_types": [
      "B"
    ],
    "type_constraints": {
      "O": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))",
        "optional(tensor(float))",
        "optional(tensor(double))",
        "optional(tensor(int64))",
        "optional(tensor(uint64))",
        "optional(tensor(int32))",
        "optional(tensor(uint32))",
        "optional(tensor(int16))",
        "optional(tensor(uint16))",
        "optional(tensor(int8))",
        "optional(tensor(uint8))",
        "optional(tensor(float16))",
        "optional(tensor(bfloat16))",
        "optional(tensor(bool))",
        "optional(tensor(string))",
        "optional(seq(tensor(float)))",
        "optional(seq(tensor(double)))",
        "optional(seq(tensor(int64)))",
        "optional(seq(tensor(uint64)))",
        "optional(seq(tensor(int32)))",
        "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(int16)))",
        "optional(seq(tensor(uint16)))",
        "optional(seq(tensor(int8)))",
        "optional(seq(tensor(uint8)))",
        "optional(seq(tensor(float16)))",
        "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(bool)))",
        "optional(seq(tensor(string)))"
      ],
      "B": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Or",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)"
      ],
      "T1": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "PRelu",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      7,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "PRelu",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "PRelu",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      16,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pad",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      2,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pad",
    "input_types": [
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pad",
    "input_types": [
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pad",
    "input_types": [
      "T",
      "tensor(int64)",
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      18,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pad",
    "input_types": [
      "T",
      "tensor(int64)",
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(bool)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ParametricSoftplus",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pow",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      7,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pow",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pow",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      14
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Pow",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      15,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "QLinearConv",
    "input_types": [
      "T1",
      "tensor(float)",
      "T1",
      "T2",
      "tensor(float)",
      "T2",
      "tensor(float)",
      "T3",
      "T4"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int8)"
      ],
      "T3": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T4": [
        "tensor(int32)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      10,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "QLinearMatMul",
    "input_types": [
      "T1",
      "tensor(float)",
      "T1",
      "T2",
      "tensor(float)",
      "T2",
      "tensor(float)",
      "T3"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int8)"
      ],
      "T3": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      10,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "QuantizeLinear",
    "input_types": [
      "T1",
      "tensor(float)",
      "T2"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(float)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      10,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "QuantizeLinear",
    "input_types": [
      "T1",
      "tensor(float)",
      "T2"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(float)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      13,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "QuantizeLinear",
    "input_types": [
      "T1",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(float16)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RNN",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T"
    ],
    "outputs_types": [
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ],
      "T1": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      7,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RNN",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T"
    ],
    "outputs_types": [
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ],
      "T1": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RandomNormal",
    "input_types": [],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RandomNormalLike",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RandomUniform",
    "input_types": [],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RandomUniformLike",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Range",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int16)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Reciprocal",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Reciprocal",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL1",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL1",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL1",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL1",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL2",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL2",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL2",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceL2",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSum",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSumExp",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSumExp",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSumExp",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceLogSumExp",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMax",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      18,
      19
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMax",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMean",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMean",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMean",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMean",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      11
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      12,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMin",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      18,
      19
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceMin",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceProd",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceProd",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceProd",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceProd",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceSum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceSum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceSum",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceSumSquare",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceSumSquare",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceSumSquare",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReduceSumSquare",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RegexFullMatch",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(string)"
      ],
      "T2": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Relu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Relu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Relu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Reshape",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      4
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Reshape",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "shape": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      5,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Reshape",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "shape": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Reshape",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "shape": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      14,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Reshape",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "shape": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Resize",
    "input_types": [
      "T",
      "tensor(float)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      10,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Resize",
    "input_types": [
      "T1",
      "T2",
      "tensor(float)",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Resize",
    "input_types": [
      "T1",
      "T2",
      "tensor(float)",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Resize",
    "input_types": [
      "T1",
      "T2",
      "tensor(float)",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      18,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Resize",
    "input_types": [
      "T1",
      "T2",
      "tensor(float)",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ReverseSequence",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      10,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RoiAlign",
    "input_types": [
      "T1",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      10,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "RoiAlign",
    "input_types": [
      "T1",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      16,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Round",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "STFT",
    "input_types": [
      "T1",
      "T2",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      17,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Scale",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScaledTanh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Scan",
    "input_types": [
      "I",
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "I": [
        "tensor(int64)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      8,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Scan",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      9,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Scan",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Scan",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      16,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Scan",
    "input_types": [
      "V"
    ],
    "outputs_types": [
      "V"
    ],
    "type_constraints": {
      "V": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Scatter",
    "input_types": [
      "T",
      "Tind",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      9,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterElements",
    "input_types": [
      "T",
      "Tind",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterElements",
    "input_types": [
      "T",
      "Tind",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterElements",
    "input_types": [
      "T",
      "Tind",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      16,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterElements",
    "input_types": [
      "T",
      "Tind",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterND",
    "input_types": [
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterND",
    "input_types": [
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterND",
    "input_types": [
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      16,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ScatterND",
    "input_types": [
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Selu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      6,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SequenceAt",
    "input_types": [
      "S",
      "I"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ],
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "I": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SequenceConstruct",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "S"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SequenceEmpty",
    "input_types": [],
    "outputs_types": [
      "S"
    ],
    "type_constraints": {
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SequenceErase",
    "input_types": [
      "S",
      "I"
    ],
    "outputs_types": [
      "S"
    ],
    "type_constraints": {
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ],
      "I": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SequenceInsert",
    "input_types": [
      "S",
      "T",
      "I"
    ],
    "outputs_types": [
      "S"
    ],
    "type_constraints": {
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ],
      "I": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SequenceLength",
    "input_types": [
      "S"
    ],
    "outputs_types": [
      "I"
    ],
    "type_constraints": {
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ],
      "I": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Shape",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Shape",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      14
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Shape",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      15,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Shape",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Shrink",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sigmoid",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sigmoid",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sign",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)"
      ]
    },
    "version_range": [
      9,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sign",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SimplifiedLayerNormalization",
    "input_types": [
      "T",
      "V"
    ],
    "outputs_types": [
      "V",
      "U"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "U": [
        "tensor(float)",
        "tensor(double)"
      ],
      "V": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sin",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sinh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Size",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(string)",
        "tensor(bool)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Size",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(string)",
        "tensor(bool)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      18
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Size",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(string)",
        "tensor(bool)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      19,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Slice",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      9
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Slice",
    "input_types": [
      "T",
      "Tind",
      "Tind",
      "Tind",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      10,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Slice",
    "input_types": [
      "T",
      "Tind",
      "Tind",
      "Tind",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Slice",
    "input_types": [
      "T",
      "Tind",
      "Tind",
      "Tind",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Softmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Softmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(double)",
        "tensor(float)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Softmax",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(double)",
        "tensor(float)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Softplus",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Softsign",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SpaceToDepth",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SpaceToDepth",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Split",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      2,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Split",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Split",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      17
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Split",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(uint64)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      18,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "SplitToSequence",
    "input_types": [
      "T",
      "I"
    ],
    "outputs_types": [
      "S"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(float16)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(string)"
      ],
      "S": [
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(int64))",
        "seq(tensor(uint64))",
        "seq(tensor(int32))",
        "seq(tensor(uint32))",
        "seq(tensor(int16))",
        "seq(tensor(uint16))",
        "seq(tensor(int8))",
        "seq(tensor(uint8))",
        "seq(tensor(float16))",
        "seq(tensor(bfloat16))",
        "seq(tensor(bool))",
        "seq(tensor(string))"
      ],
      "I": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sqrt",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sqrt",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Squeeze",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Squeeze",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Squeeze",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "StringConcat",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(string)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "StringNormalizer",
    "input_types": [
      "tensor(string)"
    ],
    "outputs_types": [
      "tensor(string)"
    ],
    "type_constraints": {
      "X": [
        "tensor(string)"
      ]
    },
    "version_range": [
      10,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "StringSplit",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2",
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(string)"
      ],
      "T2": [
        "tensor(string)"
      ],
      "T3": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      20,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sub",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      7,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sub",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      13
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sub",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      7
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      8,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Sum",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Tan",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Tanh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Tanh",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "TfIdfVectorizer",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(string)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T1": [
        "tensor(float)"
      ]
    },
    "version_range": [
      9,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ThresholdedRelu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      9
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "ThresholdedRelu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      10,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Tile",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(string)",
        "tensor(bool)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      6,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Tile",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(string)",
        "tensor(bool)"
      ],
      "T1": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "TopK",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T",
      "I"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "I": [
        "tensor(int64)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      9
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "TopK",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T",
      "I"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "I": [
        "tensor(int64)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      10,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "TopK",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T",
      "I"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(int32)"
      ],
      "I": [
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Transpose",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Transpose",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Trilu",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      14,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Unique",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T",
      "tensor(int64)",
      "tensor(int64)",
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int64)",
        "tensor(int8)",
        "tensor(string)",
        "tensor(double)"
      ]
    },
    "version_range": [
      11,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Unsqueeze",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      10
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Unsqueeze",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      11,
      12
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Unsqueeze",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ]
    },
    "version_range": [
      13,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Upsample",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      7,
      8
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Upsample",
    "input_types": [
      "T",
      "tensor(float)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int32)",
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      9,
      9
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Where",
    "input_types": [
      "B",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(string)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      9,
      15
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Where",
    "input_types": [
      "B",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(string)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      16,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "",
    "name": "Xor",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(bool)"
      ],
      "T1": [
        "tensor(bool)"
      ]
    },
    "version_range": [
      7,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "ArrayFeatureExtractor",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "Binarizer",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "CastMap",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "map(int64,tensor(string))",
        "map(int64,tensor(float))"
      ],
      "T2": [
        "tensor(float)",
        "tensor(int64)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "CategoryMapper",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(string)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(string)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "DictVectorizer",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "map(string,tensor(int64))",
        "map(string,tensor(float))",
        "map(string,tensor(double))",
        "map(int64,tensor(string))",
        "map(int64,tensor(float))",
        "map(int64,tensor(double))"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(string)",
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "FeatureVectorizer",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "Imputer",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "LabelEncoder",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(string)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(string)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      1
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "LabelEncoder",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(float)"
      ],
      "T2": [
        "tensor(string)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(float)"
      ]
    },
    "version_range": [
      2,
      3
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "LabelEncoder",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(float)",
        "tensor(string)",
        "tensor(double)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(double)",
        "tensor(double)"
      ],
      "T2": [
        "tensor(string)",
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(float)",
        "tensor(int16)",
        "tensor(string)",
        "tensor(double)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(double)"
      ]
    },
    "version_range": [
      4,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "LinearClassifier",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2",
      "tensor(float)"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(string)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "LinearRegressor",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "Normalizer",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "OneHotEncoder",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "SVMClassifier",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2",
      "tensor(float)"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "SVMRegressor",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "Scaler",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "TreeEnsembleClassifier",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2",
      "tensor(float)"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(int32)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      2
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "TreeEnsembleClassifier",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2",
      "tensor(float)"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(int32)"
      ],
      "T2": [
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)",
        "tensor(int64)",
        "tensor(string)"
      ]
    },
    "version_range": [
      3,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "TreeEnsembleRegressor",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "TreeEnsembleRegressor",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "tensor(float)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      3,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "ai.onnx.ml",
    "name": "ZipMap",
    "input_types": [
      "tensor(float)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "seq(map(string,tensor(float)))",
        "seq(map(int64,tensor(float)))"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Attention",
    "input_types": [
      "T",
      "T",
      "T",
      "M",
      "T",
      "T",
      "M"
    ],
    "outputs_types": [
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "AttnLSTM",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T",
      "T",
      "T",
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T"
    ],
    "outputs_types": [
      "T",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ],
      "T1": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "BeamSearch",
    "input_types": [
      "F",
      "I",
      "I",
      "I",
      "I",
      "T",
      "T",
      "M",
      "M",
      "I",
      "I",
      "I"
    ],
    "outputs_types": [
      "I",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "BiasGelu",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "BifurcationDetector",
    "input_types": [
      "T",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "CDist",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "ConvTransposeWithDynamicPads",
    "input_types": [
      "T",
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "CropAndResize",
    "input_types": [
      "T1",
      "T1",
      "T2",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "DequantizeLinear",
    "input_types": [
      "T1",
      "T2",
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(uint16)",
        "tensor(int16)",
        "tensor(int32)"
      ],
      "T2": [
        "tensor(float)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "DynamicQuantizeLSTM",
    "input_types": [
      "T",
      "T2",
      "T2",
      "T",
      "T1",
      "T",
      "T",
      "T",
      "T",
      "T2",
      "T",
      "T2"
    ],
    "outputs_types": [
      "T",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ],
      "T1": [
        "tensor(int32)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "DynamicQuantizeMatMul",
    "input_types": [
      "T1",
      "T2",
      "T1",
      "T2",
      "T1"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "EmbedLayerNormalization",
    "input_types": [
      "T1",
      "T1",
      "T",
      "T",
      "T",
      "T",
      "T",
      "T1",
      "T1"
    ],
    "outputs_types": [
      "T",
      "T1",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "ExpandDims",
    "input_types": [
      "T",
      "tensor(int32)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "axis": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "FastGelu",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "FusedConv",
    "input_types": [
      "T",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "FusedGemm",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "FusedMatMul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "GatherND",
    "input_types": [
      "T",
      "Tind"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int16)",
        "tensor(uint16)",
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(float16)",
        "tensor(bfloat16)",
        "tensor(bool)",
        "tensor(string)"
      ],
      "Tind": [
        "tensor(int32)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Gelu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "GreedySearch",
    "input_types": [
      "I",
      "I",
      "I",
      "T",
      "I",
      "I",
      "I"
    ],
    "outputs_types": [
      "I"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "GridSample",
    "input_types": [
      "T1",
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Inverse",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MatMulBnb4",
    "input_types": [
      "T1",
      "T2",
      "T1"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(uint8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MatMulFpQ4",
    "input_types": [
      "T1",
      "T2",
      "T3"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(uint8)"
      ],
      "T3": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MatMulInteger16",
    "input_types": [
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int16)"
      ],
      "T2": [
        "tensor(int16)"
      ],
      "T3": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MatMulIntegerToFloat",
    "input_types": [
      "T1",
      "T2",
      "T3",
      "T3",
      "T1",
      "T2",
      "T3"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int8)"
      ],
      "T3": [
        "tensor(float)",
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MatMulNBits",
    "input_types": [
      "T1",
      "T2",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)"
      ],
      "T2": [
        "tensor(uint8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MaxpoolWithMask",
    "input_types": [
      "T",
      "tensor(int32)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MultiHeadAttention",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "M",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "T",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "MurmurHash3",
    "input_types": [
      "T1"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(int32)",
        "tensor(uint32)",
        "tensor(int64)",
        "tensor(uint64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(string)"
      ],
      "T2": [
        "tensor(int32)",
        "tensor(uint32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "NGramRepeatBlock",
    "input_types": [
      "Tid",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "Tid": [
        "tensor(int64)"
      ],
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "NhwcMaxPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int8)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Pad",
    "input_types": [
      "T",
      "tensor(int64)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QAttention",
    "input_types": [
      "T1",
      "T2",
      "T3",
      "T3",
      "T3",
      "T4",
      "T1",
      "T2",
      "T3"
    ],
    "outputs_types": [
      "T3",
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T3": [
        "tensor(float)"
      ],
      "T4": [
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QEmbedLayerNormalization",
    "input_types": [
      "T1",
      "T1",
      "T2",
      "T2",
      "T2",
      "T2",
      "T2",
      "T1",
      "T",
      "T",
      "T",
      "T",
      "T",
      "T2",
      "T2",
      "T2",
      "T2",
      "T2"
    ],
    "outputs_types": [
      "T",
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QGemm",
    "input_types": [
      "TA",
      "T",
      "TA",
      "TB",
      "T",
      "TB",
      "TC",
      "T",
      "TYZ"
    ],
    "outputs_types": [
      "TY"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(float)"
      ],
      "TC": [
        "tensor(int32)",
        "tensor(int32)"
      ],
      "TA": [
        "tensor(int8)",
        "tensor(uint8)"
      ],
      "TB": [
        "tensor(int8)",
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "TYZ": [
        "tensor(int8)",
        "tensor(uint8)"
      ],
      "TY": [
        "tensor(float)",
        "tensor(int8)",
        "tensor(float)",
        "tensor(uint8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearAdd",
    "input_types": [
      "T",
      "tensor(float)",
      "T",
      "T",
      "tensor(float)",
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearAveragePool",
    "input_types": [
      "T",
      "tensor(float)",
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {},
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearConcat",
    "input_types": [
      "TF",
      "T8",
      "TV"
    ],
    "outputs_types": [
      "T8"
    ],
    "type_constraints": {},
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearConv",
    "input_types": [
      "T1",
      "tensor(float)",
      "T1",
      "T2",
      "tensor(float)",
      "T2",
      "tensor(float)",
      "T3",
      "T4"
    ],
    "outputs_types": [
      "T3"
    ],
    "type_constraints": {
      "T1": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(int8)"
      ],
      "T3": [
        "tensor(uint8)",
        "tensor(int8)"
      ],
      "T4": [
        "tensor(int32)",
        "tensor(int32)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearGlobalAveragePool",
    "input_types": [
      "T",
      "tensor(float)",
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {},
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearLeakyRelu",
    "input_types": [
      "T",
      "tensor(float)",
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearMul",
    "input_types": [
      "T",
      "tensor(float)",
      "T",
      "T",
      "tensor(float)",
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearSigmoid",
    "input_types": [
      "T",
      "tensor(float)",
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearSoftmax",
    "input_types": [
      "T",
      "tensor(float)",
      "T",
      "tensor(float)",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QLinearWhere",
    "input_types": [
      "B",
      "T",
      "TF",
      "T",
      "T",
      "TF",
      "T",
      "TF",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(uint8)",
        "tensor(int8)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QuantizeLinear",
    "input_types": [
      "T1",
      "T1",
      "T2"
    ],
    "outputs_types": [
      "T2"
    ],
    "type_constraints": {
      "T1": [
        "tensor(float)",
        "tensor(float)",
        "tensor(float)",
        "tensor(float)"
      ],
      "T2": [
        "tensor(uint8)",
        "tensor(int8)",
        "tensor(uint16)",
        "tensor(int16)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "QuickGelu",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Range",
    "input_types": [
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int16)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "RotaryEmbedding",
    "input_types": [
      "T",
      "M",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ],
      "M": [
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "SampleOp",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Sampling",
    "input_types": [
      "I",
      "I",
      "I",
      "T",
      "I",
      "I",
      "I",
      "I",
      "I"
    ],
    "outputs_types": [
      "I",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "SkipLayerNormalization",
    "input_types": [
      "T",
      "T",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "U",
      "U",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "SkipSimplifiedLayerNormalization",
    "input_types": [
      "T",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T",
      "U",
      "U",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "SparseToDenseMatMul",
    "input_types": [
      "T",
      "T1"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "sparse_tensor(float)",
        "sparse_tensor(double)",
        "sparse_tensor(int32)",
        "sparse_tensor(int64)",
        "sparse_tensor(uint32)",
        "sparse_tensor(uint64)"
      ],
      "T1": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint32)",
        "tensor(uint64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Tokenizer",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(string)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "TransposeMatMul",
    "input_types": [
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Trilu",
    "input_types": [
      "T",
      "tensor(int64)"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)",
        "tensor(double)",
        "tensor(int64)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "Unique",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T",
      "tensor(int64)",
      "tensor(int64)"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "WhisperBeamSearch",
    "input_types": [
      "F",
      "I",
      "I",
      "I",
      "I",
      "T",
      "T",
      "M",
      "M",
      "I",
      "I",
      "I",
      "I",
      "I"
    ],
    "outputs_types": [
      "I",
      "T",
      "T",
      "V",
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft",
    "name": "WordConvEmbedding",
    "input_types": [
      "T",
      "T1",
      "T1",
      "T1"
    ],
    "outputs_types": [
      "T1"
    ],
    "type_constraints": {
      "T": [
        "tensor(int32)"
      ],
      "T1": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "AveragePool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "Conv",
    "input_types": [
      "T",
      "T",
      "T",
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "GlobalAveragePool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "GlobalMaxPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "MaxPool",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "ReorderInput",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "ReorderOutput",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  },
  {
    "domain": "com.microsoft.nchwc",
    "name": "Upsample",
    "input_types": [
      "T"
    ],
    "outputs_types": [
      "T"
    ],
    "type_constraints": {
      "T": [
        "tensor(float)"
      ]
    },
    "version_range": [
      1,
      2147483647
    ],
    "execution_provider": "CPUExecutionProvider"
  }
]