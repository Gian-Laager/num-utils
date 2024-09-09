#include "nu/nu.h"
#include "pch.h"
#include <gtest/gtest.h>

TEST(Hello, World) 
{
    ASSERT_EQ(nu::add(1,2), 3);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
