/*
 * Sort Dispatch Computation Shader
 * 
 * After sorting tile_keys, this shader updates the sort dispatch parameters
 * based on the actual number of tile entries.
 */
struct TilePipelineStats {
    total_tile_entries: u32,
    visible_gaussians: u32,
    base_offset: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read_write> sort_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> indirect_dispatch: array<u32>;
@group(0) @binding(2) var<storage, read> pipeline_stats: TilePipelineStats;

// Constants from sort.ts C object
const histogram_wg_size = 256u;
const rs_histogram_block_rows = 15u;
const rs_scatter_block_rows = 15u;

@compute @workgroup_size(1)
fn update_dispatch() {
    // Get the actual count of keys to sort
    let keysize = pipeline_stats.total_tile_entries;
    
    // Compute the number of keys per workgroup
    let keys_per_workgroup = histogram_wg_size * rs_histogram_block_rows;
    
    // Compute the number of keys to sort
    let keys_count_adjusted = ((keysize + keys_per_workgroup - 1u) / keys_per_workgroup + 1u) * keys_per_workgroup;
    
    // Compute the number of scatter blocks
    let scatter_block_kvs = histogram_wg_size * rs_scatter_block_rows;
    let scatter_blocks_ru = (keysize + scatter_block_kvs - 1u) / scatter_block_kvs;
    
    // Compute the number of scatter blocks to sort
    let count_ru_scatter = scatter_blocks_ru * scatter_block_kvs;
    
    // Compute the number of histogram blocks
    let histo_block_kvs = histogram_wg_size * rs_histogram_block_rows;
    let histo_blocks_ru = (count_ru_scatter + histo_block_kvs - 1u) / histo_block_kvs;
    
    // Compute the number of histogram blocks to sort
    let count_ru_histo = histo_blocks_ru * histo_block_kvs;
    
    // Write updates into sort buffer
    sort_info[0] = keysize;
    sort_info[1] = count_ru_histo;
    sort_info[2] = 4u;
    
    // Write updates into indirect dispatch buffer
    indirect_dispatch[0] = scatter_blocks_ru;
}
