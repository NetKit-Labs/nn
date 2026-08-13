#ifndef NN_EXIT_STATUS_H
#define NN_EXIT_STATUS_H

namespace nn {

// Stable process exit codes. Commands that detect "differences" or policy
// failures use kExitDifference (1). Everything else uses a distinct class.
enum ExitStatus : int {
    kExitOk = 0,
    kExitDifference = 1,       // comparison difference / policy failure
    kExitUsage = 2,            // command-line usage error
    kExitFile = 3,             // file not found / unreadable
    kExitMalformed = 4,        // malformed model
    kExitUnsupportedFormat = 5,
    kExitUnsupportedOperator = 6,
    kExitBackendUnavailable = 7,
    kExitExecution = 8,
    kExitValidation = 9,
    kExitInternal = 10
};

}  // namespace nn

#endif
