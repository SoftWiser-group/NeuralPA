[BugLab_Wrong_Literal]^USE_ANNOTATIONS ( false ) ,^33^34^35^36^^28^38^USE_ANNOTATIONS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^AUTO_DETECT_CREATORS ( false ) ,^49^50^51^52^^44^54^AUTO_DETECT_CREATORS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^AUTO_DETECT_FIELDS ( false ) ,^64^65^66^67^^59^69^AUTO_DETECT_FIELDS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^AUTO_DETECT_GETTERS ( false ) ,^83^84^85^86^^78^88^AUTO_DETECT_GETTERS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^AUTO_DETECT_IS_GETTERS ( false ) ,^99^100^101^102^^94^104^AUTO_DETECT_IS_GETTERS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^AUTO_DETECT_SETTERS ( false ) ,^115^116^117^118^^110^120^AUTO_DETECT_SETTERS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^REQUIRE_SETTERS_FOR_GETTERS ( true ) ,^126^127^128^129^^121^131^REQUIRE_SETTERS_FOR_GETTERS ( false ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^USE_GETTERS_AS_SETTERS ( false ) ,^144^145^146^147^^139^149^USE_GETTERS_AS_SETTERS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^CAN_OVERRIDE_ACCESS_MODIFIERS ( false ) ,^156^157^158^159^^151^161^CAN_OVERRIDE_ACCESS_MODIFIERS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^INFER_PROPERTY_MUTATORS ( false ) ,^174^175^176^177^^169^179^INFER_PROPERTY_MUTATORS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^ALLOW_FINAL_FIELDS_AS_MUTATORS ( false ) ,^186^187^188^189^^181^191^ALLOW_FINAL_FIELDS_AS_MUTATORS ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^USE_STATIC_TYPING ( true ) ,^209^210^211^212^^204^214^USE_STATIC_TYPING ( false ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^DEFAULT_VIEW_INCLUSION ( false ) ,^233^234^235^236^^228^238^DEFAULT_VIEW_INCLUSION ( true ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^SORT_PROPERTIES_ALPHABETICALLY ( true ) ,^255^256^257^258^^250^260^SORT_PROPERTIES_ALPHABETICALLY ( false ) ,^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Wrong_Literal]^USE_WRAPPER_NAME_AS_PROPERTY_NAME ( true ) ;^275^276^^^^270^280^USE_WRAPPER_NAME_AS_PROPERTY_NAME ( false ) ;^[CLASS] MapperFeature   [VARIABLES] 
[BugLab_Variable_Misuse]^_defaultState = _defaultState;^281^^^^^280^282^_defaultState = defaultState;^[CLASS] MapperFeature  [METHOD] <init> [RETURN_TYPE] MapperFeature(boolean)   boolean defaultState [VARIABLES] MapperFeature  ALLOW_FINAL_FIELDS_AS_MUTATORS  AUTO_DETECT_CREATORS  AUTO_DETECT_FIELDS  AUTO_DETECT_GETTERS  AUTO_DETECT_IS_GETTERS  AUTO_DETECT_SETTERS  CAN_OVERRIDE_ACCESS_MODIFIERS  DEFAULT_VIEW_INCLUSION  INFER_PROPERTY_MUTATORS  REQUIRE_SETTERS_FOR_GETTERS  SORT_PROPERTIES_ALPHABETICALLY  USE_ANNOTATIONS  USE_GETTERS_AS_SETTERS  USE_STATIC_TYPING  USE_WRAPPER_NAME_AS_PROPERTY_NAME  boolean  _defaultState  defaultState  
[BugLab_Variable_Misuse]^public boolean enabledByDefault (  )  { return defaultState; }^285^^^^^280^290^public boolean enabledByDefault (  )  { return _defaultState; }^[CLASS] MapperFeature  [METHOD] enabledByDefault [RETURN_TYPE] boolean   [VARIABLES] MapperFeature  ALLOW_FINAL_FIELDS_AS_MUTATORS  AUTO_DETECT_CREATORS  AUTO_DETECT_FIELDS  AUTO_DETECT_GETTERS  AUTO_DETECT_IS_GETTERS  AUTO_DETECT_SETTERS  CAN_OVERRIDE_ACCESS_MODIFIERS  DEFAULT_VIEW_INCLUSION  INFER_PROPERTY_MUTATORS  REQUIRE_SETTERS_FOR_GETTERS  SORT_PROPERTIES_ALPHABETICALLY  USE_ANNOTATIONS  USE_GETTERS_AS_SETTERS  USE_STATIC_TYPING  USE_WRAPPER_NAME_AS_PROPERTY_NAME  boolean  _defaultState  defaultState  
[BugLab_Wrong_Operator]^public int getMask (  )  { return  ( 1  >>  ordinal (  )  ) ; }^288^^^^^283^293^public int getMask (  )  { return  ( 1 << ordinal (  )  ) ; }^[CLASS] MapperFeature  [METHOD] getMask [RETURN_TYPE] int   [VARIABLES] MapperFeature  ALLOW_FINAL_FIELDS_AS_MUTATORS  AUTO_DETECT_CREATORS  AUTO_DETECT_FIELDS  AUTO_DETECT_GETTERS  AUTO_DETECT_IS_GETTERS  AUTO_DETECT_SETTERS  CAN_OVERRIDE_ACCESS_MODIFIERS  DEFAULT_VIEW_INCLUSION  INFER_PROPERTY_MUTATORS  REQUIRE_SETTERS_FOR_GETTERS  SORT_PROPERTIES_ALPHABETICALLY  USE_ANNOTATIONS  USE_GETTERS_AS_SETTERS  USE_STATIC_TYPING  USE_WRAPPER_NAME_AS_PROPERTY_NAME  boolean  _defaultState  defaultState  
